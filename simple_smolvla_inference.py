#!/usr/bin/env python

import torch
import numpy as np
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    # Path to your pretrained model
    model_path = "/home/allied/Disk2/Yihao/checkpoints/smolvla/smolvla_aloha_sponge_pot/checkpoints/last/pretrained_model"
    
    # Load the pretrained policy
    print("Loading SmolVLA policy...")
    policy = SmolVLAPolicy.from_pretrained(model_path, device="cuda")
    print("Policy loaded successfully!")
    
    # Reset the policy (important before inference)
    policy.reset()
    
    # Create dummy observations (replace with your actual observations)
    batch_size = 1
    
    # Example observation batch - adjust these shapes based on your model's expected inputs
    batch = {
        # Multiple camera views (3 cameras for ALOHA)
        "observation.images.cam_high": torch.rand(batch_size, 3, 480, 640, device="cuda", dtype=torch.float32),
        "observation.images.cam_left_wrist": torch.rand(batch_size, 3, 480, 640, device="cuda", dtype=torch.float32), 
        "observation.images.cam_right_wrist": torch.rand(batch_size, 3, 480, 640, device="cuda", dtype=torch.float32),
        
        # Robot state (14-dim for ALOHA)
        "observation.state": torch.rand(batch_size, 14, device="cuda", dtype=torch.float32),
        
        # Task description (language instruction)
        "task": ["Put the sponge in the pot"]  # Replace with your task
    }
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        action = policy.select_action(batch)
    
    print(f"Predicted action shape: {action.shape}")
    print(f"Predicted action: {action}")
    
    return action

if __name__ == "__main__":
    main() 