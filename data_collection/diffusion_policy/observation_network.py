'A implementation for observation networks'

from typing import Callable, Union
import math
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
from torchvision.models import ResNet18_Weights
import numpy as np

# # pip install diffusers
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.training_utils import EMAModel


"""
================================================
Pooling Networks
================================================

implementation from - 'https://github.com/ARISE-Initiative/robomimic/blob/0ca7ce74cf8f20be32029657ef9320db033d93e9/robomimic/models/base_nets.py#L1113'
"""

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer - `Converts a feature map into a set of 2D coordinates (keypoints) 
    by computing the soft-argmax (expected position) for each channel`

    [1]. Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    
    [2]. And as per mentioned in Diffusion Policy: Visuomotor Policy Learning via Action Diffusion in section 3.2
    https://arxiv.org/pdf/2303.04137v5
    """
    
    def __init__(self, 
        input_shape, 
        num_kp=32, # can chnage according
        temperature=1.0, 
        learnable_temperature=False, 
        output_variance=False, 
        noise_std=0.0,
        ):

        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints to extract (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """

        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self.channel, self.height, self.width = input_shape     # Ensures input shape has three dimensions (C, H, W) and stores them

        # reduce the input convolution into 2D keypoints, each output channel will become one 2D keypoints
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self.channel, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else: # the convolution is already into 2D keypoints 
            self.nets = None
            self._num_kp = self.channel
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        # if learnable_temperature = True, parameter update 
        if self.learnable_temperature:
            # update the parameter, temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature) # udpate and regsiter the parameter
        else: # if learnable_temperature = False
            # fixed buffer, non-trainable 
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False) # False because we don't do update
            self.register_buffer('temperature', temperature)

            # create a normalised grid from [-1, 1]

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height)
        )

        # flattern the grid to 1D arrays of length H*W, and convert it into tensors
        pos_x = torch.from_numpy(pos_x.reshape(1, self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self.height * self.width)).float()

        # register non-learnable parameter
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None  # set it to None, so that after each iteration keypoints can get updated

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self.channel)
        return [self._num_kp, 2] # will show [K,2] because each keypoint has (x,y) co-ordinates

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self.channel)
        assert(feature.shape[2] == self.height)
        assert(feature.shape[3] == self.width)
        
        if self.nets is not None: # Applies 1×1 conv to get [B, num_kp, H, W]
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self.height * self.width) # -1 figures out the dimmension itself 
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        # if self.training: # optional only when, we want noise  
        #     noise = torch.rand_like(feature_keypoints) * self.noise_std
        #     feature_keypoints += noise

        # store a copy/cache of the keypoints for debugging for later use
        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


"""
================================================
Visual Backbone Networks
================================================

implemented CoordConv2d from - 'https://github.com/ARISE-Initiative/robomimic/blob/0ca7ce74cf8f20be32029657ef9320db033d93e9/robomimic/models/base_nets.py#L909'
implemented ResNet18 from - 'https://github.com/ARISE-Initiative/robomimic/blob/0ca7ce74cf8f20be32029657ef9320db033d93e9/robomimic/models/base_nets.py#L506'
"""

class CoordConv2d(nn.Conv2d, nn.Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented

            - Original input: `[B, C, H, W]`
            - Coordinate encoding: `[B, 2, H, W]`
            - **Concatenated**: `[B, C+2, H, W]
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[0] + 2] + input_shape[1:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == 'position':
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            # move the cache from CPU to GPU
            if self._position_enc.device != input.device:
                self._position_enc = self._position_enc.to(input.device)

            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)


class ResNet18Conv(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        # net = vision_models.resnet18(weights=(vision_models.ResNet18_Weights.DEFAULT if pretrained else None))
        net = vision_models.resnet18(weights=None) # no pre-training, let me model learn by itself.
        
        # Replace BatchNorm with GroupNorm
        self._replace_bn_with_gn(net)


        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def _replace_bn_with_gn(self, module, num_groups=32):
        """Recursively replace all BatchNorm2d layers with GroupNorm"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # Ensure num_groups divides num_features
                g = num_groups
                while child.num_features % g != 0:
                    g //= 2
                setattr(module, name, nn.GroupNorm(g, child.num_features))
            else:
                self._replace_bn_with_gn(child, num_groups)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def forward(self, input):
        return self.nets(input)


"""
================================================
For Low Level Feature Extraction (MLP)
================================================

 implemented from 'https://github.com/ARISE-Initiative/robomimic/blob/0ca7ce74cf8f20be32029657ef9320db033d93e9/robomimic/models/base_nets.py#L204'
"""
class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        layer_dims=(),
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        dropouts=None,
        normalization=False,
        output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_dims ([int]): sequence of integers for the hidden layers sizes

            layer_func: mapping per layer - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation: non-linearity per layer - defaults to ReLU

            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.

            normalization (bool): if True, apply layer normalization after each layer

            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)