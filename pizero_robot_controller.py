#!/usr/bin/env python3

import argparse
import os
import signal
import time
from functools import partial

# ALOHA imports
from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_OPEN, 
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN,
    IS_MOBILE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    enable_gravity_compensation,
    disable_gravity_compensation,
    get_arm_gripper_positions,
    ImageRecorder,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)

# Robot control imports
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rclpy

# Policy loading utilities
import torch
import cv2
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move follower arms to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [follower_bot_left, follower_bot_right],
        [start_arm_qpos] * 2,
        moving_time=4.0,
    )
    # move grippers to starting position

    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
        moving_time=0.5
    )

    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )


def get_robot_state(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS
):
    """Get current robot state (joint positions)."""
    state = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm positions
    state[:6] = follower_bot_left.core.joint_states.position[:6]
    state[7:7+6] = follower_bot_right.core.joint_states.position[:6]
    # Gripper positions - use FOLLOWER gripper normalization function
    state[6] = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(follower_bot_left.core.joint_states.position[6])
    state[7+6] = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(follower_bot_right.core.joint_states.position[6])
    return state


def follower_gripper_joint_unnormalize(normalized_value):
    """Inverse of FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN.
    
    The normalization typically maps actual gripper values to [0, 1] range.
    This function maps back from [0, 1] to actual gripper range.
    
    Based on ALOHA constants, the actual gripper range is typically around:
    - Closed: FOLLOWER_GRIPPER_JOINT_CLOSE (around 1.4)
    - Open: Some minimum value (around -0.6)
    """
    # These values are estimated based on typical ALOHA gripper ranges
    # You may need to adjust these based on your specific robot calibration
    gripper_max = FOLLOWER_GRIPPER_JOINT_OPEN  # Fully open position
    gripper_min = FOLLOWER_GRIPPER_JOINT_CLOSE   # Fully closed position (FOLLOWER_GRIPPER_JOINT_CLOSE)
    
    # Convert from [0, 1] normalized range back to actual gripper range
    return normalized_value * (gripper_max - gripper_min) + gripper_min


def apply_action(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
    action: np.ndarray,
    gripper_command: JointSingleCommand,
    use_smoothing: bool = True,
    max_joint_step: float = 0.04,
    max_gripper_step: float = 0.05,
):
    """Apply action to the robots.

    If ``use_smoothing`` is True, generate a smoothed intermediate command by
    limiting per-step movement from the current joint positions to the target
    action. This avoids large jumps by clamping the change per joint and per
    gripper at each control iteration.
    """
    # Split action into left and right arm components
    target_left_arm = action[:6]
    target_left_gripper_norm = action[6]
    target_right_arm = action[7:13]
    target_right_gripper_norm = action[13]

    if use_smoothing:
        # Current joint positions
        current_left_arm = np.array(follower_bot_left.core.joint_states.position[:6])
        current_right_arm = np.array(follower_bot_right.core.joint_states.position[:6])

        # Compute next step for arms with per-joint clamp
        left_delta = target_left_arm - current_left_arm
        right_delta = target_right_arm - current_right_arm
        left_step = np.clip(left_delta, -max_joint_step, max_joint_step)
        right_step = np.clip(right_delta, -max_joint_step, max_joint_step)
        next_left_arm = current_left_arm + left_step
        next_right_arm = current_right_arm + right_step

        # Grippers: work in normalized space; clamp per-step change
        current_left_gripper_norm = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(
            follower_bot_left.core.joint_states.position[6]
        )
        current_right_gripper_norm = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(
            follower_bot_right.core.joint_states.position[6]
        )
        left_grip_delta = target_left_gripper_norm - current_left_gripper_norm
        right_grip_delta = target_right_gripper_norm - current_right_gripper_norm
        left_grip_step = float(np.clip(left_grip_delta, -max_gripper_step, max_gripper_step))
        right_grip_step = float(np.clip(right_grip_delta, -max_gripper_step, max_gripper_step))
        next_left_gripper_norm = float(np.clip(current_left_gripper_norm + left_grip_step, 0.0, 1.0))
        next_right_gripper_norm = float(np.clip(current_right_gripper_norm + right_grip_step, 0.0, 1.0))
    else:
        next_left_arm = target_left_arm
        next_right_arm = target_right_arm
        next_left_gripper_norm = target_left_gripper_norm
        next_right_gripper_norm = target_right_gripper_norm

    # Convert grippers from normalized [0,1] to actual joint values
    left_gripper_unnormalized = follower_gripper_joint_unnormalize(next_left_gripper_norm)
    right_gripper_unnormalized = follower_gripper_joint_unnormalize(next_right_gripper_norm)

    # Apply arm actions (position control)
    follower_bot_left.arm.set_joint_positions(next_left_arm, blocking=False)
    follower_bot_right.arm.set_joint_positions(next_right_arm, blocking=False)

    # Apply gripper actions
    gripper_command.cmd = left_gripper_unnormalized
    follower_bot_left.gripper.core.pub_single.publish(gripper_command)

    gripper_command.cmd = right_gripper_unnormalized
    follower_bot_right.gripper.core.pub_single.publish(gripper_command)



def prepare_observation_batch(images: dict, robot_state: np.ndarray, task_description: str, device: str = "cuda"):
    """Prepare observation batch for SmolVLA."""
    batch = {}
    
    # Process camera images
    # Expected format: (batch_size, channels, height, width) with values in [0, 1]
    for cam_name, image in images.items():
        if image is not None:
            # Convert BGR to RGB and normalize
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Convert to tensor and normalize to [0, 1] (policy handles resizing/padding)
                image_tensor = torch.from_numpy(image_rgb).float() / 255.0
                # Rearrange to (1, C, H, W)
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Map camera names to expected format - be more flexible with naming
                cam_lower = cam_name.lower()
                if 'high' in cam_lower or 'top' in cam_lower or 'cam_high' in cam_lower:
                    batch["observation.images.cam_high"] = image_tensor.to(device)
                elif 'left' in cam_lower and ('wrist' in cam_lower or 'hand' in cam_lower):
                    batch["observation.images.cam_left_wrist"] = image_tensor.to(device)
                elif 'right' in cam_lower and ('wrist' in cam_lower or 'hand' in cam_lower):
                    batch["observation.images.cam_right_wrist"] = image_tensor.to(device)
                else:
                    # If we can't match the name, use it as the first available camera
                    if "observation.images.cam_high" not in batch:
                        batch["observation.images.cam_high"] = image_tensor.to(device)
                    elif "observation.images.cam_left_wrist" not in batch:
                        batch["observation.images.cam_left_wrist"] = image_tensor.to(device)
                    elif "observation.images.cam_right_wrist" not in batch:
                        batch["observation.images.cam_right_wrist"] = image_tensor.to(device)
    
    # Add robot state
    batch["observation.state"] = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)
    
    # Add task description
    batch["task"] = [task_description]
    
    return batch


def main(args: dict) -> None:
    model_path = args['model_path']
    task_description = args.get('task', "Put the sponge in the pot")
    save_images = args.get('save_images', False)
    save_dir = args.get('save_dir', '/tmp/robot_controller_images')
    control_frequency = args.get('control_frequency', 20.0)  # Hz
    inference_stride = int(args.get('inference_stride', 5))
    use_amp = bool(args.get('use_amp', False))
    use_smoothing = not args.get('disable_smoothing', False)
    max_joint_step = args.get('max_joint_step', 0.04)
    max_gripper_step = args.get('max_gripper_step', 0.05)
    
    # Calculate sleep duration from frequency
    control_dt = 1.0 / control_frequency
    print(f"🔄 Control frequency: {control_frequency} Hz (dt = {control_dt:.4f}s)")

    # Load policy (auto-detect type from checkpoint config)
    print("Loading policy...")
    try:
        cfg = PreTrainedConfig.from_pretrained(model_path)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(model_path)
        # Try CUDA first, fallback to CPU if RTX 5090 compatibility issue
        try:
            policy.to("cuda")
            device = "cuda"
            print("✅ Using CUDA")
        except Exception as cuda_e:
            print(f"⚠️  CUDA failed ({cuda_e}), falling back to CPU")
            policy.to("cpu")
            device = "cpu"
        
        policy.reset()
        print(f"✅ Policy loaded successfully: type='{cfg.type}'")
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        return

    # Initialize ROS node
    node = create_interbotix_global_node('aloha')
    
    # Initialize follower bots
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    
    # Initialize camera recorder
    print("Initializing cameras...")
    image_recorder = ImageRecorder(node=node, is_mobile=IS_MOBILE)
    
    # Initialize gripper command for ROS publishing
    gripper_command = JointSingleCommand()
    gripper_command.name = "gripper"  # Specify which joint we're commanding
    
    # Give cameras time to initialize
    time.sleep(1.0)

    # Initialize robots
    robot_startup(node)
    
    # Set up follower bots for control
    print("Setting up robots...")
    # torque_on(follower_bot_left)
    # torque_on(follower_bot_right)

    opening_ceremony(follower_bot_left, follower_bot_right)
    
    print("🤖 Robot control with SmolVLA started!")
    print(f"🎯 Task: {task_description}")
    print("📷 Reading camera data and controlling robots with SmolVLA")
    
    if save_images:
        print(f"💾 Camera images will be saved to: {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # Signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\nShutting down robots...")
        robot_shutdown(node)
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # Main control loop
    loop_count = 0
    loop_start_time = time.time()
    try:
        while rclpy.ok():
            iteration_start = time.time()
            
            # Get current robot state
            robot_state = get_robot_state(follower_bot_left, follower_bot_right)
            
            # Get camera images
            images = image_recorder.get_images()
            
            # Debug camera data before processing
            if loop_count % 20 == 0:  # Print debug info every 20 iterations
                elapsed_time = time.time() - loop_start_time
                actual_freq = loop_count / elapsed_time if elapsed_time > 0 else 0
                print(f"\n🔍 Debug Info - Step {loop_count} | Actual freq: {actual_freq:.1f} Hz")
                print(f"ImageRecorder camera_names: {image_recorder.camera_names}")
                print(f"Images dict keys: {list(images.keys())}")
                for cam_name, img in images.items():
                    if img is not None:
                        print(f"  {cam_name}: {img.shape} {img.dtype}")
                    else:
                        print(f"  {cam_name}: None (no data)")
            
            # Prepare observation batch for SmolVLA
            try:
                batch = prepare_observation_batch(images, robot_state, task_description, device)
                
                # Get action prediction from SmolVLA every N steps; reuse last otherwise
                if loop_count % inference_stride == 0:
                    t0 = time.time()
                    with torch.inference_mode():
                        if device == "cuda" and use_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                action_tensor = policy.select_action(batch)
                        else:
                            action_tensor = policy.select_action(batch)
                    action = action_tensor.detach().cpu().numpy().flatten()

                apply_action(
                    follower_bot_left,
                    follower_bot_right,
                    action,
                    gripper_command,
                    use_smoothing=use_smoothing,
                    max_joint_step=max_joint_step,
                    max_gripper_step=max_gripper_step,
                )
                
                # Display info every 20 iterations (about 4 times per second)
                if loop_count % 20 == 0:
                    print(f"🔄 Step {loop_count}")
                    print(f"   State: {robot_state}")
                    print(f"   Action: {action}")
                    if loop_count % inference_stride == 0:
                        inf_dt = time.time() - t0
                        print(f"   🧠 Inference time: {inf_dt:.3f}s (stride={inference_stride}, amp={'on' if (device=='cuda' and use_amp) else 'off'})")
                    
                    # Show gripper values before and after unnormalization
                    left_gripper_norm = action[6]
                    right_gripper_norm = action[13]
                    left_gripper_unnorm = follower_gripper_joint_unnormalize(left_gripper_norm)
                    right_gripper_unnorm = follower_gripper_joint_unnormalize(right_gripper_norm)
                    print(f"   🤏 Grippers: L={left_gripper_norm:.3f}→{left_gripper_unnorm:.3f}, R={right_gripper_norm:.3f}→{right_gripper_unnorm:.3f}")
                    
                    # Show current gripper positions from robot state
                    current_left_gripper = follower_bot_left.core.joint_states.position[6]
                    current_right_gripper = follower_bot_right.core.joint_states.position[6]
                    print(f"   📍 Current gripper positions: L={current_left_gripper:.3f}, R={current_right_gripper:.3f}")
                    print(f"   📏 Gripper range: CLOSE={FOLLOWER_GRIPPER_JOINT_CLOSE:.3f} to OPEN={FOLLOWER_GRIPPER_JOINT_OPEN:.3f}")
                    
                    # Camera status
                    camera_status = []
                    for cam_name in image_recorder.camera_names:
                        image = images.get(cam_name)
                        if image is not None:
                            camera_status.append(f"{cam_name}: ✅")
                        else:
                            camera_status.append(f"{cam_name}: ❌")
                    print(f"   📷 Cameras: {' | '.join(camera_status)}")
                    
                    # Optionally save images
                    if save_images and loop_count % 100 == 0:  # Save every 2 seconds
                        timestamp = time.time()
                        for cam_name, image in images.items():
                            if image is not None:
                                filename = f"{cam_name}_{timestamp:.3f}.jpg"
                                filepath = os.path.join(save_dir, filename)
                                cv2.imwrite(filepath, image)
                        print(f"   💾 Images saved to {save_dir}")
                
            except Exception as e:
                print(f"❌ Error in control loop: {e}")
                # Continue loop but don't apply action
            
            # Calculate sleep time to maintain desired frequency
            iteration_time = time.time() - iteration_start
            sleep_time = max(0, control_dt - iteration_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_count % 20 == 0:  # Warn about timing issues
                print(f"⚠️  Control loop running slow: {iteration_time:.4f}s > {control_dt:.4f}s target")
            
            loop_count += 1
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        print("Disabling robot torque...")
        # torque_off(follower_bot_left)
        # torque_off(follower_bot_right)
        robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmolVLA Robot Controller - Control ALOHA robots using SmolVLA policy')
    parser.add_argument(
        '--model_path', 
        type=str,
        required=False,
        default="/home/allied/Disk2/Yihao/checkpoints/pi_0_fast_put_sponge_into_pot/last/pretrained_model",
        help='Path to the SmolVLA pretrained model directory'
    )
    parser.add_argument(
        '--task',
        type=str,
        default="clean the dish",
        help='Task description for the robot (default: "Put the sponge in the pot")'
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save camera images to disk periodically'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/tmp/robot_controller_images',
        help='Directory to save camera images (default: /tmp/robot_controller_images)'
    )
    parser.add_argument(
        '--control_frequency',
        type=float,
        default=25.0,
        help='Desired control frequency in Hz (default: 20.0)'
    )
    parser.add_argument(
        '--inference_stride',
        type=int,
        default=5,
        help='Call policy every N control steps; reuse previous action otherwise (default: 5)'
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Enable AMP for CUDA inference'
    )
    parser.add_argument(
        '--disable_smoothing',
        action='store_true',
        help='Disable action smoothing (by default smoothing is enabled)'
    )
    parser.add_argument(
        '--max_joint_step',
        type=float,
        default=0.04,
        help='Maximum per-iteration joint change in radians (default: 0.04)'
    )
    parser.add_argument(
        '--max_gripper_step',
        type=float,
        default=0.05,
        help='Maximum per-iteration gripper change in normalized units (default: 0.05)'
    )
    
    args = vars(parser.parse_args())
    main(args) 