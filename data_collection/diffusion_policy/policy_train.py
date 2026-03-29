"""
train.py - Diffusion Policy Training
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from observation_network import SpatialSoftmax, MLP, ResNet18Conv
from dataloaders import RobosuiteDataloader
from observation_encoder import ObservationEncoder
from unet import ConditionalUnet1D  # adjust import to your filename

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Config
# ============================================================
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
action_dim = 7

num_diffusion_iters = 100
num_epochs = 200
batch_size = 32
lr = 1e-4

# ============================================================
# Dataset
# ============================================================
home_dir = os.path.expanduser("~")
dataset_path = os.path.join(
    home_dir,
    "../vla0-trl/data_collection/robosuite_human_demonstration/"
    "data_collected/1763938930_8333051/image_224.hdf5" #adjust the dataset path to the path of the dataset which you have collected
)

dataset = RobosuiteDataloader(
    dataset_path=dataset_path,
    split='train',
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# ============================================================
# Models
# ============================================================

# Observation Encoder
obs_encoder = ObservationEncoder(
    obs_shapes=OrderedDict([
        ('img_agent', (3, 224, 224)),
        ('img_hand',  (3, 224, 224)),
        ('state',     (8,)),           # 3 pos + 3 axisangle + 2 gripper
    ]),
    feature_activation=None,          
    spatial_softmax_num_kp=32,
)

obs_feature_dim = obs_encoder.output_dim     # 64 + 64 + 64 = 192
global_cond_dim = obs_horizon * obs_feature_dim  # 2 * 192 = 384

# Noise Prediction Network
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,                # 7
    global_cond_dim=global_cond_dim,     # 384
    diffusion_step_embed_dim=256,
    down_dims=[256, 512, 1024],
    kernel_size=5,
    n_groups=8,
)

# Wrap both into one ModuleDict (easy to save/load)
nets = nn.ModuleDict({
    'obs_encoder': obs_encoder,
    'noise_pred_net': noise_pred_net,
}).to(device)

# ============================================================
# Noise Scheduler + Optimizer
# ============================================================
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon',
)

optimizer = torch.optim.AdamW(nets.parameters(), lr=lr, weight_decay=1e-6)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(dataloader)
)

# ============================================================
# Training Loop
# ============================================================
"""
Each step:
  1. Encode observations → conditioning vector
  2. Sample random noise & random timestep
  3. Add noise to clean actions (forward diffusion)
  4. Predict the noise with UNet (conditioned on observations)
  5. MSE loss between predicted noise and actual noise
"""

best_loss = float('inf')
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    nets.train()
    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        # Move to GPU
        obs_dict = {
            'img_agent': batch['img_agent'].to(device),   # (B, 2, 3, 224, 224)
            'img_hand':  batch['img_hand'].to(device),     # (B, 2, 3, 224, 224)
            'state':     batch['state'].to(device),        # (B, 2, 8)
        }
        actions = batch['actions'].to(device)              # (B, 16, 7) normalized [-1, 1]
        B = actions.shape[0]

        # Step 1: Encode observations
        obs_features = nets['obs_encoder'](obs_dict)       # (B, 2, 192)
        obs_cond = obs_features.flatten(start_dim=1)       # (B, 384)

        # Step 2: Sample noise and timesteps
        noise = torch.randn_like(actions)                  # (B, 16, 7)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Step 3: Add noise to actions
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        # Step 4: Predict noise
        noise_pred = nets['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond
        )

        # Step 5: Loss and backprop
        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(nets.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Epoch summary
    avg_loss = epoch_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    # Save checkpoint
    if (epoch + 1) % 50 == 0 or avg_loss < best_loss:
        if avg_loss < best_loss:
            best_loss = avg_loss
            tag = 'best'
        else:
            tag = f'epoch_{epoch+1}'

        torch.save({
            'epoch': epoch + 1,
            'nets': nets.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            # Normalization stats needed for inference
            'action_min': dataset.action_min,
            'action_max': dataset.action_max,
            'action_range': dataset.action_range,
            'state_mean': dataset.state_mean,
            'state_std': dataset.state_std,
        }, os.path.join(save_dir, f'checkpoint_{tag}.pt'))
        print(f"  → Saved checkpoint_{tag}.pt")

print("Training complete!")

