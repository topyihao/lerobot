#!/usr/bin/env python

import cv2
import h5py
import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.datasets.video_utils import encode_video_frames


def decompress_image(compressed_data, compress_len):
    """Decompress image data using the compress_len information."""
    # This is a placeholder - you'll need to implement based on your compression format
    # Common approach is to use cv2.imdecode if images are JPEG compressed
    try:
        # Assuming JPEG compression
        img_array = np.frombuffer(compressed_data[: int(compress_len)], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Failed to decode image")
    except Exception as e:
        print(f"Error decompressing image: {e}")
        return None


def process_single_episode(hdf5_path: str, episode_idx: int):
    """Process a single HDF5 file and return the episode data."""
    print(f"Loading episode {episode_idx} from {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Extract data based on actual HDF5 structure
        observations = f["observations"]
        actions = f["action"]

        print(f"  Episode {episode_idx} - Actions shape: {actions.shape}")

        # Extract image data - these are compressed
        cam_high_compressed = observations["images"]["cam_high"][:]
        cam_left_wrist_compressed = observations["images"]["cam_left_wrist"][:]
        cam_right_wrist_compressed = observations["images"]["cam_right_wrist"][:]

        # Extract compression lengths
        compress_len = f["compress_len"][:]

        # Extract other observation data
        qpos = observations["qpos"][:]
        qvel = observations["qvel"][:]
        effort = observations["effort"][:]
        actions_data = actions[:]

        # Get dataset dimensions
        num_timesteps = qpos.shape[0]

        # Find actual episode length (remove padding/zeros at the end)
        episode_length = num_timesteps
        for t in range(num_timesteps - 1, -1, -1):
            if np.any(actions_data[t] != 0):
                episode_length = t + 1
                break

        print(
            f"  Episode {episode_idx} - Actual length: {episode_length}/{num_timesteps}"
        )

        return {
            "cam_high_compressed": cam_high_compressed,
            "cam_left_wrist_compressed": cam_left_wrist_compressed,
            "cam_right_wrist_compressed": cam_right_wrist_compressed,
            "compress_len": compress_len,
            "qpos": qpos,
            "qvel": qvel,
            "effort": effort,
            "actions_data": actions_data,
            "episode_length": episode_length,
        }


def convert_aloha_folder_to_lerobot(
    folder_path: str,
    repo_id: str,
    task_description: str,
    output_dir: str = "data",
    fps: int = 50,
    max_workers: int = 8,  # Increased default workers for faster processing
    file_pattern: str = "episode_*.hdf5",
    overwrite: bool = False,
):
    """
    Convert a folder of ALOHA HDF5 files to LeRobot format.

    Args:
        folder_path: Path to the folder containing HDF5 files
        repo_id: Repository ID (e.g., "username/dataset_name")
        task_description: Description of the task
        output_dir: Output directory for the dataset
        fps: Frames per second
        max_workers: Number of workers for concurrent processing
        file_pattern: Pattern to match HDF5 files (e.g., "episode_*.hdf5")
        overwrite: If True, remove existing dataset directory first
    """
    
    # Monkey-patch encode_video_frames to use H.264 instead of AV1 for better torchcodec compatibility
    original_encode_video_frames = encode_video_frames
    def patched_encode_video_frames(imgs_dir, video_path, fps, vcodec="h264", pix_fmt="yuv420p", g=1, crf=28, fast_decode=0, log_level=None, overwrite=False):
        # Force H.264 codec with faster settings for speed
        print(f"Encoding video with codec: {vcodec} (optimized for speed)")
        return original_encode_video_frames(
            imgs_dir=imgs_dir, 
            video_path=video_path, 
            fps=fps, 
            vcodec=vcodec,  # H.264 for torchcodec compatibility
            pix_fmt=pix_fmt, 
            g=g,  # Smaller keyframe interval for faster encoding
            crf=crf,  # Higher CRF = lower quality but faster encoding
            fast_decode=fast_decode, 
            log_level=log_level, 
            overwrite=overwrite
        )
    
    # Apply the patch globally
    import lerobot.datasets.video_utils as video_utils_module
    video_utils_module.encode_video_frames = patched_encode_video_frames
    print("Patched video encoding to use H.264 instead of AV1 for better torchcodec compatibility")

    # Validate repo_id format
    check_repo_id(repo_id)

    # Handle existing directory
    output_path = Path(output_dir)
    if output_path.exists():
        if overwrite:
            import shutil

            print(f"Removing existing directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            # Create a new directory with timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{output_dir}_{timestamp}"
            print(f"Directory exists, using: {output_dir}")

    # Find all HDF5 files in the folder
    folder_path = Path(folder_path)
    hdf5_files = sorted(list(folder_path.glob(file_pattern)))

    if not hdf5_files:
        raise ValueError(
            f"No HDF5 files found in {folder_path} matching pattern {file_pattern}"
        )

    print(f"Found {len(hdf5_files)} HDF5 files to convert:")
    for i, file_path in enumerate(hdf5_files):
        print(f"  {i}: {file_path.name}")

    # Process first file to get image dimensions and data structure
    first_episode_data = process_single_episode(str(hdf5_files[0]), 0)

    # Decompress first image to get dimensions
    print("Decompressing sample image to get dimensions...")
    sample_img = decompress_image(
        first_episode_data["cam_high_compressed"][0],
        first_episode_data["compress_len"][0, 0],
    )
    if sample_img is None:
        raise ValueError("Failed to decompress sample image. Check compression format.")

    img_height, img_width, img_channels = sample_img.shape
    print(f"Image dimensions: {img_height}x{img_width}x{img_channels}")

    # Define dataset features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (first_episode_data["qpos"].shape[-1],),
            "names": [
                f"joint_{i}" for i in range(first_episode_data["qpos"].shape[-1])
            ],
        },
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (img_height, img_width, img_channels),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": (img_height, img_width, img_channels),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (img_height, img_width, img_channels),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (first_episode_data["actions_data"].shape[-1],),
            "names": [
                f"action_{i}"
                for i in range(first_episode_data["actions_data"].shape[-1])
            ],
        },
    }

    # Create LeRobot dataset
    print(f"Creating LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="aloha",
        root=output_dir,
        use_videos=True,
    )

    # Start image writer for efficient video processing with batch encoding
    dataset.start_image_writer(
        num_processes=0, num_threads=max_workers  # Use threads only
    )
    
    # Use batch encoding for much faster video processing (encode 10 episodes at once)
    dataset.batch_encoding_size = 10

    # Process all episodes
    total_frames = 0
    for episode_idx, hdf5_file in enumerate(hdf5_files):
        print(
            f"\n=== Processing Episode {episode_idx + 1}/{len(hdf5_files)}: {hdf5_file.name} ==="
        )

        # Load episode data
        if episode_idx == 0:
            # We already loaded the first episode
            episode_data = first_episode_data
        else:
            episode_data = process_single_episode(str(hdf5_file), episode_idx)

        episode_length = episode_data["episode_length"]

        # Process frames for this episode
        for frame_idx in range(episode_length):
            if frame_idx % 200 == 0 or frame_idx == episode_length - 1:
                print(f"  Processing frame {frame_idx + 1}/{episode_length}")

            # Decompress images for this frame
            cam_high_img = decompress_image(
                episode_data["cam_high_compressed"][frame_idx],
                episode_data["compress_len"][0, frame_idx],
            )
            cam_left_wrist_img = decompress_image(
                episode_data["cam_left_wrist_compressed"][frame_idx],
                episode_data["compress_len"][1, frame_idx],
            )
            cam_right_wrist_img = decompress_image(
                episode_data["cam_right_wrist_compressed"][frame_idx],
                episode_data["compress_len"][2, frame_idx],
            )

            if any(
                img is None
                for img in [cam_high_img, cam_left_wrist_img, cam_right_wrist_img]
            ):
                print(f"  Warning: Failed to decompress images for frame {frame_idx}")
                continue

            frame_data = {
                "observation.state": episode_data["qpos"][frame_idx],
                "observation.images.cam_high": cam_high_img,
                "observation.images.cam_left_wrist": cam_left_wrist_img,
                "observation.images.cam_right_wrist": cam_right_wrist_img,
                "action": episode_data["actions_data"][frame_idx],
            }

            dataset.add_frame(frame_data, task=task_description)

        # Save the episode
        dataset.save_episode()
        total_frames += episode_length
        print(f"  Episode {episode_idx + 1} completed: {episode_length} frames")

    # Push to hub (commented out to skip)
    # print("Pushing dataset to hub...")
    # dataset.push_to_hub(tags=["aloha", "robotics"], private=True)

    print("\n=== Conversion Complete ===")
    print(
        f"Successfully converted {len(hdf5_files)} episodes with {total_frames} total frames!"
    )
    print(f"Dataset saved locally at: {output_dir}")
    return dataset


# Usage example
if __name__ == "__main__":
    # Example usage - convert entire folder
    convert_aloha_folder_to_lerobot(
        folder_path="/home/aloha/Disk2/aloha_dataset/aloha_open_zipper_bag",
        repo_id="local/aloha_open_zipper_bag",
        task_description=(
            "Use the left gripper to pick up and hold the zipper bag. Use the right gripper to grasp the zipper and slide it to the right to open the bag."
        ),
        output_dir="/home/aloha/Disk2/lerobot_dataset/aloha_open_zipper_bag",  # H.264 version
        fps=50,
        max_workers=6,  # Use more workers for faster conversion
        file_pattern="episode_*.hdf5",
    )
