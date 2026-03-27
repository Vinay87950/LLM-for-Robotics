# LLM-for-Robotics: VLA-0 Reimplementation with TRL

A reimplementation of [VLA-0](https://github.com/NVlabs/vla0) using [TRL](https://github.com/huggingface/trl)'s SFTTrainer, adapted for my own robotics project.

Based on the minimal [vla0-trl](https://github.com/MilkClouds/vla0-trl) codebase (~1,200 lines), which fine-tunes Qwen2.5-VL to predict actions as text. No custom architecture needed.

> **Note:** The video demonstration and the dataset link will be updated shortly. Please check back soon, and full code is not availabe yet, working on it, it will also be updated shortly.

## Installation

We recommend using [`uv`](https://docs.astral.sh/uv/) for managing dependencies.

```bash
uv venv --python 3.11
uv pip install -e .
# LeRobot dependency
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb

# For evaluation
uv pip install -e ".[eval]"

# Do not forget activating your venv
source .venv/bin/activate
```

## Usage

### Train

```bash
# Single GPU
python scripts/train.py --config configs/vla0.yaml

# Multi-GPU
accelerate launch --num_processes=8 scripts/train.py --config configs/vla0.yaml
```

### Eval

```bash
python scripts/eval.py \
    --model_path ./runs/vla0/checkpoint-xxx \
    --task_suite libero_spatial \
    --action_horizon 8 \
    --ensemble_prediction 8 \
    --torch_compile \
    --skip_evaluated \
    --shard_id 0 --num_shards 10
```

| Argument | Description |
|----------|-------------|
| `--task_suite` | Task suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--action_horizon` | Execute N actions before re-querying model (default: 1) |
| `--ensemble_prediction` | Average N overlapping action chunks (default: 1 = off) |
| `--torch_compile` | Enable torch.compile for faster inference |
| `--skip_evaluated` | Skip episodes with existing result videos |
| `--shard_id`, `--num_shards` | Parallelize: run shard M of N (e.g., 0/10, 1/10, ...) |
| `--log_dir` | Output directory (default: auto-generated with timestamp) |

Note: When running multiple shards in parallel, specify `--log_dir` explicitly to ensure all shards write to the same directory.

### SLURM

For SLURM users, see [`scripts/train.sbatch`](scripts/train.sbatch) and [`scripts/eval.sbatch`](scripts/eval.sbatch).

## Configuration

See [`configs/vla0.yaml`](configs/vla0.yaml). Key parameters:

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 4e-5 (5e-6 × 8 GPUs) |
| `num_train_epochs` | 32 |
| `per_device_train_batch_size` | 8 |
| `horizon` | 8 |

## Project Structure

```
├── configs/vla0.yaml       # Training config
├── scripts/
│   ├── train.py            # Training entry
│   └── eval.py             # Evaluation entry
└── src/
    ├── rv_train/           # Dataset, collator, model
    └── rv_eval/            # Evaluator
```

## Attribution

This is a derivative work of [VLA-0](https://github.com/NVlabs/vla0) by NVIDIA, based on the minimal reimplementation by [vla0-trl](https://github.com/MilkClouds/vla0-trl).

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use this code, please cite the original VLA-0 paper:

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and Hadfield, Hugo and Yang, Xuning and Blukis, Valts and Ramos, Fabio},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```

And the vla0-trl reimplementation:

```bibtex
@misc{vla0-trl,
  author = {Suhwan Choi},
  title = {vla0-trl: Minimal VLA-0 Reimplementation with TRL},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/MilkClouds/vla0-trl},
  doi = {10.5281/ZENODO.18712424}
}
```
