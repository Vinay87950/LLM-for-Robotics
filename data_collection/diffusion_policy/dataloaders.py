""" This file contains the dataloader for the robosuite environment. """

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
import os 
import math


"""
    This function is to load data and make it ready for the training process by using some transformation for image.
    As this data contains is in the form of hdf5 file which contains following specifications which is just Mujoco physics.

    ***Actions in 7D format (likely 3D position + 3D rotation + 1D gripper)***.

    For example
    
    data/
│   ├── demo_1/
│   │   ├── action_dict/
│   │   │   ├── gripper  (186, 1) (float32)
│   │   │   ├── rel_pos  (186, 3) (float32)
│   │   │   ├── rel_rot_6d  (186, 6) (float32)
│   │   │   ├── rel_rot_axis_angle  (186, 3) (float32)
│   │   ├── actions  (186, 7) (float64)
│   │   ├── dones  (186,) (int64)
│   │   ├── obs/
│   │   │   ├── agentview_image  (186, 224, 224, 3) (uint8)
│   │   │   ├── object  (186, 10) (float64)
│   │   │   ├── robot0_eef_pos  (186, 3) (float64)
│   │   │   ├── robot0_eef_quat  (186, 4) (float64)
│   │   │   ├── robot0_eef_quat_site  (186, 4) (float32)
│   │   │   ├── robot0_eye_in_hand_image  (186, 224, 224, 3) (uint8)
│   │   │   ├── robot0_gripper_qpos  (186, 2) (float64)
│   │   │   ├── robot0_gripper_qvel  (186, 2) (float64)
│   │   │   ├── robot0_joint_pos  (186, 7) (float64)
│   │   │   ├── robot0_joint_pos_cos  (186, 7) (float64)
│   │   │   ├── robot0_joint_pos_sin  (186, 7) (float64)
│   │   │   ├── robot0_joint_vel  (186, 7) (float64)
│   │   ├── rewards  (186,) (float64)
│   │   ├── states  (186, 32) (float64)
"""


def quat2axisangle(quat):
    """
    Converts quaternion(s) to axis-angle format (vectorized).
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): Can be either:
            - Single quaternion: (4,) array as (x,y,z,w)
            - Multiple quaternions: (T, 4) array as (x,y,z,w)

    Returns:
        np.array: 
            - Single: (3,) axis-angle exponential coordinates
            - Multiple: (T, 3) axis-angle exponential coordinates
    """
    # Handle both single quaternion and array of quaternions
    single_quat = (quat.ndim == 1)
    if single_quat:
        quat = quat.reshape(1, -1)
    
    # Clip quaternion w component
    q = quat.copy()
    q[:, 3] = np.clip(q[:, 3], -1.0, 1.0)
    
    # Calculate denominator
    den = np.sqrt(1.0 - q[:, 3] * q[:, 3])
    
    # Handle near-zero rotations
    eps = 1e-6
    result = np.zeros((q.shape[0], 3))
    valid_mask = den > eps
    
    # Only compute for valid (non-zero) rotations
    if valid_mask.any():
        result[valid_mask] = (q[valid_mask, :3] * 2.0 * np.arccos(q[valid_mask, 3])[:, None]) / den[valid_mask, None]
    
    # Return shape matching input
    if single_quat:
        return result[0]
    return result


class RobosuiteDataloader(Dataset):
    """
    Dataloader for Robosuite Pick-and-Place with multiple camera views.
    
    State: Robot proprioception (EEF pose + gripper) - 8D
    Vision: Let the network learn object information from images
    Actions: 7-DOF (3 pos + 3 rot + 1 gripper)

    """

    def __init__(self, dataset_path, split='train', pred_horizon=16, obs_horizon=2, action_horizon=8):
        self.pred_horizon = pred_horizon    
        self.obs_horizon = obs_horizon      
        self.action_horizon = action_horizon 
        self.split = split
        
        #  Define Image Transforms 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #  Load Data 
        self.all_data = []
        self.indices = [] 
        
        print(f"Loading '{split}' data from {dataset_path}...")
        
        all_actions_list = []
        all_states_list = []

        with h5py.File(dataset_path, 'r') as f:
            # Determine which demos to load
            if 'mask' in f and split in f['mask']:
                demo_keys = [k.decode('utf-8') if isinstance(k, bytes) else k 
                            for k in f['mask'][split][:]]
            else:
                print(f"Warning: No mask/{split} found. Loading all demos.")
                demo_keys = list(f['data'].keys())

            print(f"Found {len(demo_keys)} demos for split '{split}'")

            #  Load each demo
            for demo_key in demo_keys:
                demo = f['data'][demo_key]
                
                # Load BOTH camera views
                img_agent = demo['obs']['agentview_image'][:]           # Third-person view
                img_hand = demo['obs']['robot0_eye_in_hand_image'][:]   # Wrist camera
                
                # Robot State: ONLY proprioceptive info (what the robot "feels")
                eef_pos = demo['obs']['robot0_eef_pos'][:]              # (T, 3)
                eef_quat = quat2axisangle(demo['obs']['robot0_eef_quat'][:]) # (T, 3)
                gripper_qpos = demo['obs']['robot0_gripper_qpos'][:]    # (T, 2)
                
                # State vector: EEF pose + gripper = 5D (since we don't need orientation)
                # State describes pose
                # The network should learn about objects from vision!
                state = np.concatenate([
                    eef_pos,        # End effector position (3)
                    eef_quat,       # End effector orientation (3)
                    gripper_qpos,   # Gripper joint positions (2)
                ], axis=-1)
                
                # Actions: 7D (3 pos + 3 rot + 1 gripper)
                # Action describes the movement (like how should the robot move)
                actions = demo['actions'][:] 

                # Validate
                total_steps = actions.shape[0]
                assert img_agent.shape[0] == total_steps
                assert img_hand.shape[0] == total_steps
                assert state.shape[0] == total_steps
                
                # Store
                self.all_data.append({
                    'img_agent': img_agent,
                    'img_hand': img_hand,
                    'state': state,
                    'action': actions
                })
                
                all_actions_list.append(actions)
                all_states_list.append(state)

                # Create sliding window indices
                for t in range(total_steps - self.pred_horizon + 1):
                    self.indices.append((len(self.all_data) - 1, t))

        # Calculate Normalization Stats 
        all_actions = np.concatenate(all_actions_list, axis=0)
        all_states = np.concatenate(all_states_list, axis=0)
        
        # Action normalization: Min-Max to [-1, 1]
        self.action_min = torch.from_numpy(all_actions.min(axis=0)).float()
        self.action_max = torch.from_numpy(all_actions.max(axis=0)).float()
        
        self.action_range = self.action_max - self.action_min
        self.action_range = torch.where(
            self.action_range < 1e-6, 
            torch.ones_like(self.action_range), 
            self.action_range
        )
        
        # State normalization: Standardization (mean=0, std=1)
        self.state_mean = torch.from_numpy(all_states.mean(axis=0)).float()
        self.state_std = torch.from_numpy(all_states.std(axis=0)).float()
        
        self.state_std = torch.where(
            self.state_std < 1e-6,
            torch.ones_like(self.state_std),
            self.state_std
        )

        print(f" Loaded {len(self.indices)} samples from {len(self.all_data)} demos")
        print(f" Action dim: {all_actions.shape[1]}, State dim: {all_states.shape[1]}")
        print(f" Action range per dim:")
        for i in range(all_actions.shape[1]):
            print(f"    Dim {i}: [{self.action_min[i]:.4f}, {self.action_max[i]:.4f}]")

    def normalize_action(self, action):
        """Rescale action from [min, max] to [-1, 1]"""
        action = (action - self.action_min) / self.action_range
        action = action * 2 - 1
        return action

    def unnormalize_action(self, action):
        """Rescale action from [-1, 1] to [min, max]"""
        action = (action + 1) / 2
        action = action * self.action_range + self.action_min
        return action
    
    def normalize_state(self, state):
        """Standardize state to mean=0, std=1"""
        return (state - self.state_mean) / self.state_std
    
    def unnormalize_state(self, state):
        """Restore original state scale"""
        return state * self.state_std + self.state_mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_idx, start_t = self.indices[idx]
        demo = self.all_data[demo_idx]
        
        #  Get Action Chunk (Target)
        end_t = start_t + self.pred_horizon
        actions_raw = torch.from_numpy(demo['action'][start_t:end_t]).float()
        actions = self.normalize_action(actions_raw)

        #  Get Observations (Input) 
        # Stack past frames for temporal context
        obs_indices = []
        for i in range(self.obs_horizon):
            t = start_t - (self.obs_horizon - 1) + i
            t = max(0, t)  # Clamp to 0 for episode start
            obs_indices.append(t)

        img_agent_list = []
        img_hand_list = []
        state_list = []
        
        for t in obs_indices:
            # Process both camera views
            img_a = self.transform(demo['img_agent'][t])
            img_agent_list.append(img_a)
            
            img_h = self.transform(demo['img_hand'][t])
            img_hand_list.append(img_h)
            
            # Process state (proprioception only)
            s = torch.from_numpy(demo['state'][t]).float()
            s = self.normalize_state(s)
            state_list.append(s)
            
        # Stack over temporal dimension
        img_agent = torch.stack(img_agent_list)  # (obs_horizon, 3, 224, 224)
        img_hand = torch.stack(img_hand_list)    # (obs_horizon, 3, 224, 224)
        state = torch.stack(state_list)          # (obs_horizon, 8)

        return {
            'img_agent': img_agent,   # (obs_horizon, 3, 224, 224)
            'img_hand': img_hand,     # (obs_horizon, 3, 224, 224)
            'state': state,           # (obs_horizon, 8) - EEF + gripper only
            'actions': actions        # (pred_horizon, 7)
        }


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    dataset_path = os.path.join(home_dir, "../vla0-trl/data_collection/robosuite_human_demonstration/data_collected/1763938930_8333051/image_224.hdf5")

    
    train_dataset = RobosuiteDataloader(
        dataset_path=dataset_path,
        split='train',
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print("\n=== Batch Shapes ===")
    print(f"Agent view: {batch['img_agent'].shape}")   # (B, obs_horizon, 3, 224, 224)
    print(f"Hand view:  {batch['img_hand'].shape}")    # (B, obs_horizon, 3, 224, 224)
    print(f"States:     {batch['state'].shape}")       # (B, obs_horizon, 8)
    print(f"Actions:    {batch['actions'].shape}")     # (B, pred_horizon, 7)
    
 