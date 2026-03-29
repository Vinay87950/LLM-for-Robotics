"""
The script for evaluating trained Diffusion Policy.
"""
import argparse
import json
import h5py
import imageio
import numpy as np
import os
from collections import OrderedDict, deque
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import robosuite as suite
from robosuite.wrappers import Wrapper

from observation_network import SpatialSoftmax, MLP, ResNet18Conv
from dataloaders import RobosuiteDataloader, quat2axisangle 
from observation_encoder import ObservationEncoder
from unet import ConditionalUnet1D


class DiffusionPolicyAgent:
    def __init__(self, ckpt_path, device):
        self.device = device
        
        # Hyperparameters 
        self.pred_horizon = 16
        self.obs_horizon = 2
        self.action_horizon = 8
        self.action_dim = 7
        
        # 1. Load Checkpoint
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 2. Extract Normalization Stats
        self.stats = {
            'action_min': checkpoint['action_min'].to(device),
            'action_range': checkpoint['action_range'].to(device),
            'state_mean': checkpoint['state_mean'].to(device),
            'state_std': checkpoint['state_std'].to(device)
        }
        
        # 3. Initialize Networks
        obs_encoder = ObservationEncoder(
            obs_shapes=OrderedDict([
                ('img_agent', (3, 224, 224)),
                ('img_hand',  (3, 224, 224)),
                ('state',     (8,)),
            ]),
            feature_activation=None,
            spatial_softmax_num_kp=32,
        )
        global_cond_dim = self.obs_horizon * obs_encoder.output_dim
        
        unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
        )
        
        self.nets = nn.ModuleDict({
            'obs_encoder': obs_encoder,
            'noise_pred_net': unet,
        }).to(device)
        self.nets.load_state_dict(checkpoint['nets'])
        self.nets.eval()
        
        # 4. Initialize Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, # Match training
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # 5. Vision Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.reset()

    def reset(self):
        """Clear action queue and observation history on episode reset"""
        self.obs_queue = deque(maxlen=self.obs_horizon)
        self.action_queue = deque()

    def _process_obs(self, raw_obs):
        """Extract and format raw robosuite obs into network inputs"""
        img_agent = self.transform(raw_obs['agentview_image']).to(self.device)
        img_hand = self.transform(raw_obs['robot0_eye_in_hand_image']).to(self.device)
        
        eef_pos = raw_obs['robot0_eef_pos']
        eef_quat = quat2axisangle(raw_obs['robot0_eef_quat'])
        gripper_qpos = raw_obs['robot0_gripper_qpos']
        
        state = np.concatenate([eef_pos, eef_quat, gripper_qpos])
        state = torch.from_numpy(state).float().to(self.device)
        # Normalize state
        state = (state - self.stats['state_mean']) / self.stats['state_std']
        
        return img_agent, img_hand, state

    @torch.no_grad()
    def get_action(self, raw_obs):
        """Returns the next action, running diffusion if the queue is empty"""
        # Update obs history
        img_agent, img_hand, state = self._process_obs(raw_obs)
        self.obs_queue.append((img_agent, img_hand, state))
        
        # Pad queue if episode just started
        while len(self.obs_queue) < self.obs_horizon:
            self.obs_queue.append((img_agent, img_hand, state))
            
        if len(self.action_queue) == 0:
            # --- We need to replan ---
            # 1. Stack obs history
            batch_img_agent = torch.stack([x[0] for x in self.obs_queue]).unsqueeze(0) # (1, obs_horizon, C, H, W)
            batch_img_hand = torch.stack([x[1] for x in self.obs_queue]).unsqueeze(0)
            batch_state = torch.stack([x[2] for x in self.obs_queue]).unsqueeze(0)
            
            obs_dict = {'img_agent': batch_img_agent, 'img_hand': batch_img_hand, 'state': batch_state}
            
            # 2. Encode conditioning
            obs_features = self.nets['obs_encoder'](obs_dict)
            global_cond = obs_features.flatten(start_dim=1)
            
            # 3. Run Diffusion
            noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            self.noise_scheduler.set_timesteps(100) # num inference steps
            
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.nets['noise_pred_net'](sample=noisy_action, timestep=k, global_cond=global_cond)
                noisy_action = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_action).prev_sample
            
            # 4. Unnormalize actions and populate queue
            actions = noisy_action.squeeze(0) # (16, 7)
            actions = (actions + 1) / 2
            actions = actions * self.stats['action_range'] + self.stats['action_min']
            actions = actions.cpu().numpy()
            
            # Take only the first `action_horizon` steps (Receding Horizon Control)
            self.action_queue.extend(actions[:self.action_horizon])
            
        return self.action_queue.popleft()


def run_trained_agent(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DiffusionPolicyAgent(args.agent, device)

    # Initialize Environment

    env = suite.make(
        env_name=args.env if args.env else "Lift",
        robots="Panda",
        has_renderer=args.render,
        has_offscreen_renderer=(args.video_path is not None),
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224, 
        camera_widths=224,  
        reward_shaping=True,
        control_freq=20,
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if args.video_path is not None:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    rollout_stats = []
    
    for ep_idx in range(args.n_rollouts):
        obs = env.reset()
        agent.reset()
        
        total_reward = 0.
        success = False
        step = 0
        
        print(f"Starting Rollout {ep_idx+1}/{args.n_rollouts}")
        for step in range(args.horizon):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            if env._check_success():
                success = True
            
            # Visualization / Video Recording
            if args.render:
                env.render()
            if video_writer is not None and step % args.video_skip == 0:
                frames = [env.sim.render(camera_name=cam, height=512, width=512)[::-1] for cam in args.camera_names]
                video_writer.append_data(np.concatenate(frames, axis=1))
                
            if done or success:
                break
                
        rollout_stats.append({
            'Return': total_reward, 
            'Horizon': step + 1, 
            'Success': float(success)
        })
        print(f"  Rollout {ep_idx+1} finished. Success: {success}, Return: {total_reward:.2f}")

    if video_writer is not None:
        video_writer.close()
        print(f"Video saved to {args.video_path}")

    # Print summary
    avg_success = np.mean([s['Success'] for s in rollout_stats])
    avg_return = np.mean([s['Return'] for s in rollout_stats])
    print("\n=== Evaluation Summary ===")
    print(f"Total Rollouts: {args.n_rollouts}")
    print(f"Average Success Rate: {avg_success*100:.1f}%")
    print(f"Average Return: {avg_return:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint .pt file")
    parser.add_argument("--n_rollouts", type=int, default=10, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=300, help="maximum horizon per rollout")
    parser.add_argument("--env", type=str, default="Lift", help="robosuite environment name")
    parser.add_argument("--render", action='store_true', help="enable on-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="path to output mp4 video")
    parser.add_argument("--video_skip", type=int, default=1, help="render frames to video every n steps")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview"], help="camera(s) for rendering")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    
    args = parser.parse_args()
    run_trained_agent(args)