#!/usr/bin/env python3
"""
Convert a LeRobot v3 dataset (with video-based images) to LIBERO-compatible
LeRobot v2.0 format suitable for vla0-trl training.

follow this - 'https://github.com/huggingface/lerobot/blob/f6b16f6d97155e3ce34ab2a1ec145e9413588197/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py'

Transformations applied:
  - Rename keys: action → actions, observation.state → state,
    observation.images.agentview → image, observation.images.wrist → wrist_image
  - FPS downsample: 20 → 10 (every 2nd frame)
  - Images: decoded from MP4 video → stored as inline PIL images in parquet
  - Image resize: 224×224 → 256×256 (LIBERO standard; training code resizes to 224)

Usage:
    python data_collection/convert_to_lerobot.py \
        --input /home/jovyan/vla0/my_code/data_conversion/lerobot_output_pick_cube \  
        --output ./libero_pick_place \
        --target-fps 10 --target-size 256 
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import av
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Video frame extraction
# ─────────────────────────────────────────────────────────────────────

def decode_video_frames(video_path: Path) -> list:
    """Decode all frames from an MP4 video file.

    Returns:
        List of PIL.Image objects in RGB format.
    """
    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        img = frame.to_image()  # PIL.Image in RGB
        frames.append(img)
    container.close()
    return frames


def get_hf_features(target_size: int) -> datasets.Features:
    """Define HuggingFace dataset features with proper Image type."""
    return datasets.Features({
        "image": datasets.Image(),
        "wrist_image": datasets.Image(),
        "state": datasets.Sequence(datasets.Value("float32"), length=8),
        "actions": datasets.Sequence(datasets.Value("float32"), length=7),
        "timestamp": datasets.Value("float32"),
        "frame_index": datasets.Value("int64"),
        "episode_index": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "task_index": datasets.Value("int64"),
    })


# ─────────────────────────────────────────────────────────────────────
# Episode mapping: which frames belong to which episode
# ─────────────────────────────────────────────────────────────────────

def build_episode_frame_map(df: pd.DataFrame) -> dict:
    """Build a mapping from episode_index to list of (global_row_index, frame_index).

    Returns:
        {episode_index: [(row_idx_in_df, frame_index), ...]}
    """
    episodes = {}
    for idx, row in df.iterrows():
        ep = int(row["episode_index"])
        fi = int(row["frame_index"])
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append((idx, fi))

    # Sort each episode by frame_index
    for ep in episodes:
        episodes[ep].sort(key=lambda x: x[1])

    return episodes


# ─────────────────────────────────────────────────────────────────────
# Find the video file that contains frames for a given episode
# ─────────────────────────────────────────────────────────────────────

def find_video_for_episode(
    video_base_dir: Path,
    camera_key: str,
    episode_index: int,
    info: dict,
) -> Path:
    """Locate the video file for a given camera and episode.

    The v3.0 format stores videos as:
      videos/{video_key}/chunk-{chunk}/file-{file}.mp4

    We need to figure out which chunk/file the episode belongs to.
    """
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunks_size
    # In v3.0, all episodes in one chunk share one video file per camera
    # The file index corresponds to the episode chunk grouping
    file_idx = 0  # Single file per chunk in this dataset

    video_path = video_base_dir / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    return video_path


# ─────────────────────────────────────────────────────────────────────
# Build the episode frame offsets from episode metadata
# ─────────────────────────────────────────────────────────────────────

def load_episode_metadata(input_dir: Path) -> pd.DataFrame:
    """Load episode metadata parquet to get frame offsets per episode."""
    meta_episodes_dir = input_dir / "meta" / "episodes"
    all_meta = []
    for chunk_dir in sorted(meta_episodes_dir.iterdir()):
        if chunk_dir.is_dir():
            for pq_file in sorted(chunk_dir.glob("*.parquet")):
                df = pd.read_parquet(pq_file)
                all_meta.append(df)
    if all_meta:
        return pd.concat(all_meta, ignore_index=True)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# Decode all video frames for a camera (entire video file)
# ─────────────────────────────────────────────────────────────────────

def decode_all_video_frames_for_camera(
    input_dir: Path,
    camera_key: str,
    info: dict,
) -> list:
    """Decode ALL frames from the video file for a given camera.

    Returns list of PIL Images indexed by global frame position in the video.
    """
    # For the dataset structure with chunk-000/file-000.mp4
    video_dir = input_dir / "videos"
    video_file = video_dir / camera_key / "chunk-000" / "file-000.mp4"

    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    logger.info("Decoding video frames from: %s", video_file)
    return decode_video_frames(video_file)


# ─────────────────────────────────────────────────────────────────────
# Stats computation (LIBERO v2.0 uses global stats)
# ─────────────────────────────────────────────────────────────────────

def compute_stats(all_actions: np.ndarray, all_states: np.ndarray) -> dict:
    """Compute global min/max/mean/std statistics for actions and states."""
    return {
        "actions": {
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        },
        "state": {
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────

def convert(
    input_dir: Path,
    output_dir: Path,
    target_fps: int = 10,
    target_size: int = 256,
    dry_run: bool = False,
):
    """Convert LeRobot v3 dataset to LIBERO-compatible v2.0 format."""

    # ── Load source info ──────────────────────────────────────────
    with open(input_dir / "meta" / "info.json") as f:
        src_info = json.load(f)

    src_fps = src_info["fps"]
    downsample_ratio = src_fps // target_fps
    logger.info("Source FPS: %d → Target FPS: %d (downsample every %d frames)",
                src_fps, target_fps, downsample_ratio)

    # ── Load source parquet data ──────────────────────────────────
    data_dir = input_dir / "data"
    all_parquets = sorted(data_dir.rglob("*.parquet"))
    logger.info("Found %d parquet files", len(all_parquets))

    dfs = [pd.read_parquet(pq) for pq in all_parquets]
    src_df = pd.concat(dfs, ignore_index=True)
    logger.info("Total source frames: %d", len(src_df))

    # ── Load tasks ────────────────────────────────────────────────
    tasks_pq = input_dir / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_pq)
    task_map = {}
    for task_name, row in tasks_df.iterrows():
        task_map[int(row["task_index"])] = str(task_name)
    logger.info("Tasks: %s", task_map)

    # ── Build episode frame mapping ───────────────────────────────
    episode_map = build_episode_frame_map(src_df)
    total_episodes = len(episode_map)
    logger.info("Total episodes: %d", total_episodes)

    # ── Decode video frames for both cameras ──────────────────────
    logger.info("Decoding agentview video...")
    agentview_frames = decode_all_video_frames_for_camera(
        input_dir, "observation.images.agentview", src_info
    )
    logger.info("  → decoded %d frames", len(agentview_frames))

    logger.info("Decoding wrist video...")
    wrist_frames = decode_all_video_frames_for_camera(
        input_dir, "observation.images.wrist", src_info
    )
    logger.info("  → decoded %d frames", len(wrist_frames))

    if dry_run:
        logger.info("[DRY RUN] Showing sample conversion for episode 0:")
        ep0_frames = episode_map[0]
        logger.info("  Episode 0 has %d frames at %d FPS", len(ep0_frames), src_fps)
        downsampled = ep0_frames[::downsample_ratio]
        logger.info("  After downsampling to %d FPS: %d frames", target_fps, len(downsampled))
        row_idx, frame_idx = ep0_frames[0]
        logger.info("  Sample action (frame 0): %s", src_df.iloc[row_idx]["action"])
        logger.info("  Sample state (frame 0): %s", src_df.iloc[row_idx]["observation.state"])
        logger.info("  Agentview image size: %s", agentview_frames[0].size)
        logger.info("  Target image size: %dx%d", target_size, target_size)
        return

    # ── Clean output dir ──────────────────────────────────────────
    if output_dir.exists():
        logger.warning("Removing existing output: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ── Build a cumulative frame offset map for video indexing ────
    # Video frames are stored sequentially: ep0_frame0, ep0_frame1, ..., ep1_frame0, ...
    # We need to map (episode_index, frame_within_episode) → video_frame_index
    # The video contains ALL frames across episodes concatenated
    episode_video_offsets = {}
    cum_offset = 0
    for ep_idx in sorted(episode_map.keys()):
        episode_video_offsets[ep_idx] = cum_offset
        cum_offset += len(episode_map[ep_idx])

    # ── Convert each episode ──────────────────────────────────────
    all_rows = []
    all_actions_list = []
    all_states_list = []
    new_global_idx = 0
    chunks_size = 1000

    for ep_idx in tqdm(sorted(episode_map.keys()), desc="Converting episodes"):
        ep_frames = episode_map[ep_idx]
        video_offset = episode_video_offsets[ep_idx]
        task_idx = int(src_df.iloc[ep_frames[0][0]]["task_index"])
        task_str = task_map.get(task_idx, "pick up the cube from the table")

        # Downsample: take every Nth frame
        downsampled_frames = ep_frames[::downsample_ratio]

        for new_frame_idx, (src_row_idx, src_frame_idx) in enumerate(downsampled_frames):
            src_row = src_df.iloc[src_row_idx]

            # Get video frame index (position within the concatenated video)
            frame_pos_in_episode = ep_frames.index((src_row_idx, src_frame_idx))
            video_frame_idx = video_offset + frame_pos_in_episode

            # Get and resize images (keep as PIL for HF datasets)
            agentview_img = agentview_frames[video_frame_idx]
            wrist_img = wrist_frames[video_frame_idx]

            if target_size != agentview_img.size[0]:
                agentview_img = agentview_img.resize(
                    (target_size, target_size), Image.LANCZOS
                )
                wrist_img = wrist_img.resize(
                    (target_size, target_size), Image.LANCZOS
                )

            action = np.array(src_row["action"], dtype=np.float32)
            state = np.array(src_row["observation.state"], dtype=np.float32)

            all_actions_list.append(action)
            all_states_list.append(state)

            row = {
                "image": agentview_img,
                "wrist_image": wrist_img,
                "state": state.tolist(),
                "actions": action.tolist(),
                "timestamp": float(new_frame_idx) / target_fps,
                "frame_index": new_frame_idx,
                "episode_index": ep_idx,
                "index": new_global_idx,
                "task_index": task_idx,
            }
            all_rows.append(row)
            new_global_idx += 1

    logger.info("Total converted frames: %d (from %d source frames)",
                len(all_rows), len(src_df))

    # ── Write parquet files using HF datasets (proper Image encoding) ──
    total_frames = len(all_rows)
    hf_features = get_hf_features(target_size)

    # Split into chunks of episodes
    ep_indices = sorted(set(r["episode_index"] for r in all_rows))
    chunk_episodes = []
    for i in range(0, len(ep_indices), chunks_size):
        chunk_episodes.append(ep_indices[i:i + chunks_size])

    for chunk_idx, chunk_eps in enumerate(chunk_episodes):
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Write per-episode parquet files using HF datasets
        for ep_idx in tqdm(chunk_eps, desc=f"Writing chunk {chunk_idx}"):
            ep_rows = [r for r in all_rows if r["episode_index"] == ep_idx]

            # Transpose list-of-dicts → dict-of-lists for HF datasets
            ep_data = {
                "image": [r["image"] for r in ep_rows],
                "wrist_image": [r["wrist_image"] for r in ep_rows],
                "state": [r["state"] for r in ep_rows],
                "actions": [r["actions"] for r in ep_rows],
                "timestamp": [r["timestamp"] for r in ep_rows],
                "frame_index": [r["frame_index"] for r in ep_rows],
                "episode_index": [r["episode_index"] for r in ep_rows],
                "index": [r["index"] for r in ep_rows],
                "task_index": [r["task_index"] for r in ep_rows],
            }

            ds = datasets.Dataset.from_dict(ep_data, features=hf_features)
            ep_file = chunk_dir / f"episode_{ep_idx:06d}.parquet"
            ds.to_parquet(str(ep_file))

    logger.info("Wrote parquet files to: %s/data/", output_dir)

    # ── Write meta/info.json ──────────────────────────────────────
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_map),
        "total_videos": 0,
        "total_chunks": len(chunk_episodes),
        "chunks_size": chunks_size,
        "fps": target_fps,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {
                "dtype": "image",
                "shape": [target_size, target_size, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [target_size, target_size, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [7],
                "names": ["actions"],
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ── Write meta/stats.json ─────────────────────────────────────
    all_actions_arr = np.array(all_actions_list)
    all_states_arr = np.array(all_states_list)
    stats = compute_stats(all_actions_arr, all_states_arr)

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Write meta/tasks.jsonl ────────────────────────────────────
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_idx, task_str in sorted(task_map.items()):
            f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")

    # ── Write meta/episodes.jsonl (required by lerobot) ───────────
    # Count frames per episode after downsampling
    episode_frame_counts = {}
    episode_tasks = {}
    for r in all_rows:
        ep = r["episode_index"]
        episode_frame_counts[ep] = episode_frame_counts.get(ep, 0) + 1
        episode_tasks[ep] = task_map.get(r["task_index"], "pick up the cube from the table")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_idx in sorted(episode_frame_counts.keys()):
            ep_entry = {
                "episode_index": ep_idx,
                "length": episode_frame_counts[ep_idx],
                "task": episode_tasks[ep_idx],
                "task_index": 0,
            }
            f.write(json.dumps(ep_entry) + "\n")

    logger.info("Wrote episodes.jsonl with %d episodes", len(episode_frame_counts))

    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info("  Output dir     : %s", output_dir)
    logger.info("  Total episodes : %d", total_episodes)
    logger.info("  Total frames   : %d (from %d at %d FPS)", total_frames, len(src_df), src_fps)
    logger.info("  Target FPS     : %d", target_fps)
    logger.info("  Image size     : %dx%d", target_size, target_size)
    logger.info("  Format         : LeRobot v2.0 (LIBERO-compatible)")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 dataset to LIBERO-compatible v2.0 format.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to source LeRobot v3 dataset (e.g., final_data_cleaned/)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for converted dataset",
    )
    parser.add_argument(
        "--target-fps", type=int, default=10,
        help="Target FPS after downsampling (default: 10)",
    )
    parser.add_argument(
        "--target-size", type=int, default=256,
        help="Target image size (default: 256, matching LIBERO)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show sample conversion without writing output",
    )
    args = parser.parse_args()

    convert(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_fps=args.target_fps,
        target_size=args.target_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
