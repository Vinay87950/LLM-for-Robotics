#!/usr/bin/env python3
"""VLA-0 Evaluation Script for Robosuite benchmark.

Usage:
    python scripts/eval.py \
        --model_path /home/jovyan/vla0-trl/runs/vla0_sft_lora_pickcube_second_trial/final \
        --num_episodes 3 \
        --action_horizon 2 \
        --ensemble_prediction 8 \
        --save_video
"""

import argparse
from datetime import datetime
from pathlib import Path

from rv_eval.evaluator import RobosuiteEvaluator
from rv_eval.robosuite_env import build_robosuite_env
from rv_train.model import QwenVLActor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLA-0 on Robosuite")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, default=None)
    
    # Environment
    parser.add_argument("--env_name", type=str, default="Lift")
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--task_string", type=str, default="pick up the cube from the table")
    parser.add_argument("--camera_resolution", type=int, default=256)
    parser.add_argument("--control_freq", type=int, default=10)

    # Evaluation settings
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--action_horizon", type=int, default=1)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    
    parser.add_argument("--ensemble_prediction", type=int, default=1)
    parser.add_argument("--ensemble_version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ensemble_weight", type=float, default=0.5)

    # Model params
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--num_bins", type=int, default=1000)
    parser.add_argument("--torch_compile", action="store_true")

    # Image processing
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    parser.add_argument("--tile_images", action="store_true", default=True)
    parser.add_argument("--no_tile", dest="tile_images", action="store_false")

    # Output
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="save_video", action="store_false")
    parser.add_argument("--save_debug_images", action="store_true")
    parser.add_argument("--debug_actions", action="store_true")

    return parser.parse_args()


def build_log_dir(args, timestamp: str) -> str:
    model_name = (
        Path(Path(args.model_path).parent.name) / Path(args.model_path).name
        if "checkpoint-" in args.model_path
        else Path(args.model_path).name
    )
    return str(Path("eval_logs") / f"robosuite_{model_name}" / timestamp)


def main():
    args = parse_args()

    # Auto-detect stats path
    stats_path = args.stats_path
    if stats_path is None:
        model_dir = Path(args.model_path).parent
        candidate = model_dir / "dataset_stats.json"
        if candidate.exists():
            stats_path = str(candidate)
        else:
            candidate = model_dir.parent / "dataset_stats.json"
            if candidate.exists():
                stats_path = str(candidate)

    if stats_path is None:
        raise ValueError("Could not find dataset_stats.json. Specify --stats_path")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or build_log_dir(args, timestamp)

    print(f"Loading model from: {args.model_path}")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Task String: {args.task_string}")

    model = QwenVLActor(
        model_path=args.model_path,
        stats_path=stats_path,
        horizon=args.horizon,
        action_dim=args.action_dim,
        num_bins=args.num_bins,
        torch_compile=args.torch_compile,
    )

    env = build_robosuite_env(
        env_name=args.env_name,
        robot=args.robot,
        camera_resolution=args.camera_resolution,
        control_freq=args.control_freq,
        action_dim=args.action_dim,
    )

    evaluator = RobosuiteEvaluator(
        model=model,
        log_dir=log_dir,
        save_video=args.save_video,
        seed=args.seed,
        action_horizon=args.action_horizon,
        frame_skip=args.frame_skip,
        img_size=args.img_size,
        crop_ratio=args.crop_ratio,
        tile_images=args.tile_images,
        ensemble_prediction=args.ensemble_prediction,
        ensemble_version=args.ensemble_version,
        ensemble_weight=args.ensemble_weight,
        save_debug_images=args.save_debug_images,
        debug_actions=args.debug_actions,
    )

    print("Starting evaluation...")
    evaluator.evaluate(
        env=env,
        num_episodes=args.num_episodes,
        task_name=args.task_string,
        max_steps=args.max_steps,
    )
    
    env.close()
    print(f"\nResults saved to: {log_dir}/")

if __name__ == "__main__":
    main()
