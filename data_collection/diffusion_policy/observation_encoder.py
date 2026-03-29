""" This file contains the observation encoder network. """
import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader #basic data loader 
from collections import OrderedDict
from observation_network import SpatialSoftmax, MLP, ResNet18Conv
from dataloaders import RobosuiteDataloader  #data loader for robosuite environment
from torchinfo import summary

"""
The process is as follows :
Step	        || Data Shape	        || Meaning
---------------------------------------------------------------------------------------
Input	        || (32, 2, 3, 224, 224)	|| Image (Batch, time_horizon, RGB, Height, Width)
ResNet	        || (64, 512, 7, 7)	    || "There are edges and textures here"
SpatialSoftmax	|| (64, 32, 2)	        || "Important feature #1 is at (x=0.2, y=0.5)"
Flatten Vision	|| (64, 64)	            || coordinate list for 32 features
Concatenate	    || (64, 192)	        || Combined understanding of the world
Output	        || (32, 2, 192)	        || The Embedding used for Diffusion
"""
class ObservationEncoder(nn.Module):
    def __init__(self, obs_shapes, feature_activation=nn.ReLU, spatial_softmax_num_kp=32):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.obs_nets = nn.ModuleDict()
        self.activation = feature_activation() if feature_activation else None
        self.num_kp = spatial_softmax_num_kp

        for key, shape in obs_shapes.items():
            if key.startswith('img_'):
                C, H, W = shape
                # Vision Backbone
                resnet = ResNet18Conv(input_channel=C, pretrained=False, input_coord_conv=True)
                # Dynamic shape check
                with torch.no_grad():
                    dummy = torch.zeros(1, C, H, W)
                    feat_map_shape = resnet(dummy).shape[1:]
                
                self.obs_nets[key] = nn.Sequential(
                    resnet,
                    SpatialSoftmax(feat_map_shape, num_kp=spatial_softmax_num_kp)
                )
            elif key == 'state':
                self.obs_nets[key] = MLP(
                    input_dim=shape[0], output_dim=64, 
                    layer_dims=[128, 128], normalization=True
                )

    def forward(self, obs_dict):
        feats = []
        # Get B (Batch) and T (Time/Horizon)
        any_tensor = next(iter(obs_dict.values()))
        B, T = any_tensor.shape[:2]

        for key in self.obs_shapes.keys():
            x = obs_dict[key] # (B, T, ...)
            # Fold Time: (B*T, ...)
            x = x.reshape(B * T, *x.shape[2:])
            
            # Forward
            if self.obs_nets[key] is not None:
                x = self.obs_nets[key](x)
                # if self.activation is not None: x = self.activation(x)
            
            # Flatten & Unfold Time
            x = torch.flatten(x, start_dim=1)
            x = x.reshape(B, T, -1)
            feats.append(x)
        
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self):
        dim = 0
        for k in self.obs_shapes:
            if k.startswith('img_'): dim += 2 * self.num_kp
            elif k == 'state': dim += 64
        return dim

def create_obs_encoder_for_robosuite():
    return ObservationEncoder(
        obs_shapes=OrderedDict([
            ('img_agent', (3, 224, 224)),
            ('img_hand', (3, 224, 224)),
            ('state', (8,)) # 3 pos + 3 rot + 2 gripper
        ]),
        spatial_softmax_num_kp=32
    )


if __name__ == "__main__":
    print(f"--- Running on {torch.cuda.get_device_name(0)} ---")
    
    home_dir = os.path.expanduser("~")
    # load dataset
    dataset_path = os.path.join(
        home_dir, 
        "../vla0-trl/data_collection/robosuite_human_demonstration/data_collected/1763938930_8333051/image_224.hdf5"
    )
 
    train_dataset = RobosuiteDataloader(
        dataset_path=dataset_path,
        split='train',
        pred_horizon=16, # future steps
        obs_horizon=2,   # Note: Horizon is 2 (TIME), two frames at a time 
        action_horizon=8 # keep only first 8 steps 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize Model on GPU
    obs_encoder = create_obs_encoder_for_robosuite().to(device)



    # Get One Real Batch
    print("Fetching batch from dataloader...")
    batch = next(iter(train_loader))
    
    #  pass the whole (B, T) sequence.
    obs_dict = {
        'img_agent': batch['img_agent'].to(device, non_blocking=True),
        'img_hand':  batch['img_hand'].to(device, non_blocking=True),
        'state':     batch['state'].to(device, non_blocking=True),
    }
    
    #  Run Encoder
    print("Encoding observations...")
    features = obs_encoder(obs_dict)
        
    print(f"\n=== Encoded Features (Output) ===")
    print(f"Features shape: {features.shape}")
    
    # Expected: (Batch, Obs_Horizon, Feature_Dim)
    # Dim = (32 kp * 2) + (32 kp * 2) + 64 = 192
    expected_shape = (32, 2, 192)
    
    if features.shape == expected_shape:
        print(f"Output matches expected shape {expected_shape}")
    else:
        print(f"Expected {expected_shape}, got {features.shape}")

    print(f"\n=== Model Info ===")
    trainable_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")