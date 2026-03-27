"""Robosuite evaluation loop for VLA-0."""

import csv
import logging
import os
import time
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("evaluator")

DUMMY_ACTION = [0.0] * 6 + [-1.0]

def flip_image(img: np.ndarray) -> np.ndarray:
    """Robosuite and LIBERO images may be flipped; flip them back if needed."""
    return np.ascontiguousarray(img[::-1, ::-1])

def preprocess_obs(
    obs: Dict,
    img_size: int = 224,
    crop_ratio: float = 0.675,
    tile_images: bool = True,
) -> Image.Image:
    """Preprocess observation for model input."""
    cams = ["agentview_image", "robot0_eye_in_hand_image"]
    images = []

    for cam in cams:
        img = flip_image(obs[cam])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if crop_ratio < 1.0:
            h, w = img.shape[-2:]
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            img = TF.crop(img, top, left, crop_h, crop_w)

        if img_size > 0:
            img = TF.resize(img, [img_size, img_size])

        img = (img * 255).byte()
        img = img.permute(1, 2, 0).numpy()
        images.append(img)

    if tile_images:
        tiled = np.concatenate(images, axis=1)
        return Image.fromarray(tiled)

    return [Image.fromarray(img) for img in images]


class RobosuiteEvaluator:
    """Evaluator for VLA models on Robosuite benchmark."""

    def __init__(
        self,
        model,
        log_dir: str = "./eval_logs",
        save_video: bool = True,
        seed: int = 7,
        action_horizon: int = 1,
        frame_skip: int = 10,
        img_size: int = 224,
        crop_ratio: float = 0.675,
        tile_images: bool = True,
        ensemble_prediction: int = 1,
        ensemble_version: int = 1,
        ensemble_weight: float = 0.5,
        save_debug_images: bool = False,
        debug_actions: bool = False,
    ):
        self.model = model
        self.log_dir = log_dir
        self.save_video = save_video
        self.seed = seed
        self.action_horizon = action_horizon
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.ensemble_prediction = ensemble_prediction
        self.ensemble_version = ensemble_version
        self.ensemble_weight = ensemble_weight
        self.save_debug_images = save_debug_images
        self.debug_actions = debug_actions

        os.makedirs(log_dir, exist_ok=True)

    def _save_debug_images(self, obs: Dict, processed: Image.Image, ep: int) -> None:
        d = os.path.join(self.log_dir, "debug_images")
        os.makedirs(d, exist_ok=True)
        cams = ["agentview_image", "robot0_eye_in_hand_image"]
        for cam in cams:
            if cam in obs:
                raw = obs[cam].copy()
                Image.fromarray(raw).save(os.path.join(d, f"ep{ep}_raw_{cam}.png"))
                Image.fromarray(flip_image(raw)).save(os.path.join(d, f"ep{ep}_flipped_{cam}.png"))
        if isinstance(processed, Image.Image):
            processed.save(os.path.join(d, f"ep{ep}_model_input.png"))

    def run_episode(
        self,
        env,
        instruction: str,
        max_steps: int,
        episode_idx: int = 0,
    ) -> tuple:
        """Run a single evaluation episode."""
        obs = env.reset()

        frames = []
        action_chunk = None
        action_i = 0
        action_horizon = self.action_horizon

        old_action_chunks = [] if self.ensemble_prediction > 1 else None
        
        debug_saved = False
        all_actions = []
        model_queries = 0

        for t in tqdm(range(max_steps + self.frame_skip), desc="steps", leave=False):
            if t < self.frame_skip:
                obs, reward, done, info = env.step(DUMMY_ACTION)
                continue

            if action_chunk is None or action_i >= action_horizon:
                image = preprocess_obs(obs, self.img_size, self.crop_ratio, self.tile_images)
                
                if self.save_debug_images and not debug_saved:
                    self._save_debug_images(obs, image, episode_idx)
                    debug_saved = True

                action_chunk = self.model.predict(image, instruction).numpy()
                model_queries += 1

                if self.debug_actions:
                    logger.info("  Query %d raw actions:\n%s", 
                                model_queries, 
                                np.array2string(action_chunk, precision=4, suppress_small=True))

                # Ensemble prediction
                if self.ensemble_prediction > 1 and old_action_chunks is not None:
                    old_action_chunks.append(action_chunk.copy())
                    if len(old_action_chunks) > self.ensemble_prediction:
                        old_action_chunks.pop(0)

                    if len(old_action_chunks) > 1:
                        ensemble_chunk = np.zeros_like(action_chunk)
                        ensemble_count = np.zeros_like(action_chunk)
                        new_old_chunks = []
                        num_old = len(old_action_chunks)
                        for i, old_chunk in enumerate(old_action_chunks[:-1]):
                            if len(old_chunk) <= action_horizon:
                                continue
                            old_chunk = old_chunk[action_horizon:]
                            new_old_chunks.append(old_chunk)

                            weight = self.ensemble_weight if self.ensemble_version == 1 else self.ensemble_weight ** (num_old - i - 1)
                            ensemble_chunk[: len(old_chunk)] += weight * old_chunk
                            ensemble_count[: len(old_chunk)] += weight

                        new_old_chunks.append(old_action_chunks[-1])
                        ensemble_chunk += old_action_chunks[-1]
                        ensemble_count += 1

                        old_action_chunks = new_old_chunks
                        action_chunk = ensemble_chunk / ensemble_count

                action_i = 0
                action_horizon = min(self.action_horizon, len(action_chunk))

            act = action_chunk[action_i].tolist()
            act[-1] = 1.0 if act[-1] > 0 else -1.0
            all_actions.append(np.array(act))

            obs, reward, done, info = env.step(act)
            if "agentview_image" in obs:
                frames.append(flip_image(obs["agentview_image"]))
            action_i += 1

            # Unified success condition matching user's request
            is_success_call = env._check_success() if hasattr(env, "_check_success") else False
            if is_success_call or reward > 0 or done:
                self._log_action_summary(all_actions, model_queries)
                return True, frames, len(all_actions)

        self._log_action_summary(all_actions, model_queries)
        return False, frames, len(all_actions)

    def _log_action_summary(self, actions: List[np.ndarray], queries: int) -> None:
        if not actions:
            return
        acts = np.array(actions)
        logger.info("  Action stats (%d steps, %d queries):", len(acts), queries)
        logger.info("    pos  mean=%s  std=%s",
                    np.array2string(acts[:, :3].mean(0), precision=4),
                    np.array2string(acts[:, :3].std(0), precision=4))
        logger.info("    rot  mean=%s  std=%s",
                    np.array2string(acts[:, 3:6].mean(0), precision=4),
                    np.array2string(acts[:, 3:6].std(0), precision=4))
        logger.info("    grip mean=%.3f  %%open=%.1f%%",
                    acts[:, -1].mean(), (acts[:, -1] > 0).mean() * 100)

        if acts[:, :3].std(0).max() < 0.001:
            logger.warning("    ⚠  Position actions near-zero variance — model may be stuck!")

    def _append_csv(self, task: str, run_idx: int, success: bool, steps: int, wall_s: float):
        """Append result to CSV."""
        csv_path = os.path.join(self.log_dir, "results.csv")
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["task", "episode", "success", "steps", "wall_s"])
            writer.writerow([task, run_idx, int(success), steps, round(wall_s, 2)])

    @torch.no_grad()
    def evaluate(
        self,
        env,
        num_episodes: int,
        task_name: str,
        max_steps: int,
    ) -> Dict:
        """Evaluate wrapper running multiple episodes."""
        results = {"success": 0, "failure": 0, "episodes": []}
        t0 = time.time()

        for ep in range(num_episodes):
            logger.info("Episode %d/%d", ep + 1, num_episodes)
            t_ep = time.time()
            
            try:
                success, frames, steps = self.run_episode(env, task_name, max_steps, ep)
            except Exception as e:
                logger.error("Episode %d crashed: %s", ep, str(e))
                success, frames, steps = False, [], 0

            dt = time.time() - t_ep
            
            if success:
                results["success"] += 1
            else:
                results["failure"] += 1

            sr = results["success"] / (ep + 1) * 100
            logger.info("  %s | %d steps | %.1fs | SR %.1f%%",
                        "✓ SUCCESS" if success else "✗ FAILURE", steps, dt, sr)

            results["episodes"].append({
                "episode": ep, 
                "success": success, 
                "steps": steps, 
                "wall_s": round(dt, 2)
            })
            
            self._append_csv(task_name, ep, success, steps, dt)

            if self.save_video and frames:
                suffix = "success" if success else "failure"
                video_path = f"{self.log_dir}/ep_{ep:03d}_{suffix}.mp4"
                try:
                    imageio.mimwrite(video_path, frames, fps=10)
                except Exception as ve:
                    logger.warning("Failed to save video: %s", str(ve))

        elapsed = time.time() - t0
        rate = results["success"] / num_episodes * 100 if num_episodes > 0 else 0
        
        self._print_results(results["success"], num_episodes, rate, elapsed)
        return results

    def _print_results(self, successes: int, total: int, rate: float, elapsed: float):
        """Print evaluation results."""
        G, B, RST = "\033[92m", "\033[1m", "\033[0m"
        logger.info("=" * 60)
        logger.info("RESULT  %d / %d  (%.1f%%)  in %.1fs", successes, total, rate, elapsed)
        logger.info("=" * 60)
        print(f"{B}{'=' * 60}{RST}")
        print(f"{B}TOTAL SUCCESS: {G}{successes}{RST}/{total} ({rate:.1f}%){RST}")
        print(f"{B}{'=' * 60}{RST}")
