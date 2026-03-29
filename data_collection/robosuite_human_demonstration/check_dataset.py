import h5py
import json
import os
import numpy as np

# Path setup
home_dir = os.path.expanduser("~")
# change the dataset path to the path of the dataset you want to check, after you collect the dataset using the collect_human_demonstration.py script
dataset_path = os.path.join(home_dir, "../vla0-trl/data_collection/robosuite_human_demonstration/data_collected/1763938930_8333051/image_224.hdf5")

# Configure NumPy to print ALL values without truncation
np.set_printoptions(
    threshold=np.inf,       # Print all elements, no matter how many
    linewidth=200,          # Wider lines before wrapping
    precision=4,            # 4 decimal places
    suppress=True           # Suppress scientific notation for small numbers
)

def print_row(name, obj):
    """Helper function to print items with indentation"""
    depth = name.count("/")
    indent = "│   " * depth
    base_name = name.split("/")[-1]
    
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}├── {base_name}  {obj.shape} ({obj.dtype})")
    else:
        print(f"{indent}├── {base_name}/")

# --- Main Execution ---
if not os.path.exists(dataset_path):
    print(f" File not found at: {dataset_path}")
else:
    with h5py.File(dataset_path, "r") as f:
        print(f"=== HDF5 Structure: {os.path.basename(dataset_path)} ===")
        
        # 1. Automatically visit every node in the file
        f.visititems(print_row)

        # 2. Simplified Controller/Env Info
        print("\n" + "="*40)
        print("=== CONFIGURATION ===")
        if "data" in f and "env_args" in f["data"].attrs:
            env_args = json.loads(f["data"].attrs["env_args"])
            print(json.dumps(env_args, indent=2)) 
        else:
            print("No environment metadata found.")
        
        # 3. Print ALL values from states and actions
        print("\n" + "="*40)
        print("=== DATA VALUES - ALL TIMESTEPS ===")
        
        demo = f["data/demo_16"]
        
        # Print ALL actions
        print("\n--- Actions (all 186 timesteps) ---")
        actions = demo["actions"][:]
        print(f"Shape: {actions.shape}")
        print(actions)
        
        # Print ALL states
        print("\n--- States (all 186 timesteps) ---")
        states = demo["states"][:]
        print(f"Shape: {states.shape}")
        print(states)