# LLM-for-Robotics: VLA-0 Reimplementation with TRL

A reimplementation of [VLA-0](https://github.com/NVlabs/vla0) using [TRL](https://github.com/huggingface/trl)'s SFTTrainer, adapted for my own robotics project.

Based on the minimal [vla0-trl](https://github.com/MilkClouds/vla0-trl) codebase (~1,200 lines), which fine-tunes Qwen2.5-VL to predict actions as text. No custom architecture needed.

> **Note:** The dataset link will be updated shortly.
> **Video:** [results](https://drive.google.com/drive/folders/1piHjM_snGT9zLn19ja3NrnNEfKosHbea?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)

## Installation

We recommend using [`uv`](https://docs.astral.sh/uv/) for managing dependencies.

```bash
uv venv --python 3.11
uv pip install -e .
# LeRobot dependency
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb

# For evaluation (can skip this, it's only needen when you train on official dataset 'https://huggingface.co/datasets/physical-intelligence/libero'
uv pip install -e ".[eval]"

# Do not forget activating your venv
source .venv/bin/activate
```

## Usage

### Train

```bash
# Single GPU
python scripts/train.py --config configs/my_config.yaml

```

### Eval

```bash
python scripts/eval.py \
    --model_path ./runs/vla0/checkpoint-xxx \
    --action_horizon 8 \
    --ensemble_prediction 8 \
    --save_video
```

| Argument | Description |
|----------|-------------|
| `--action_horizon` | Execute N actions before re-querying model (default: 1) |
| `--ensemble_prediction` | Average N overlapping action chunks (default: 1 = off) |



## Configuration

See [`configs/my_config.yaml`](configs/my_config.yaml). Key parameters:

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 4e-5 |
| `num_train_epochs` | 20 |
| `per_device_train_batch_size` | 8 |
| `horizon` | 8 |

## Project Structure

```
├── configs/
│   └── my_config.yaml                          # Training configuration
├── data_collection/
│   ├── convert_to_libero_format.py             # Dataset format conversion
│   ├── diffusion_policy/                       # Diffusion policy implementation
│   │   ├── dataloaders.py
│   │   ├── observation_encoder.py
│   │   ├── observation_network.py
│   │   ├── policy_network.py
│   │   ├── policy_train.py
│   │   └── rollouts.py
│   └── robosuite_human_demonstration/          # Human demo collection
│       ├── collect_human_demonstration.py
│       └── check_dataset.py
├── scripts/
│   ├── train.py                                # Training entry point
│   └── eval.py                                 # Evaluation entry point
└── src/
    ├── rv_train/                               # Training pipeline
    │   ├── collator.py                         # Data collator for VLA
    │   ├── dataset.py                          # Dataset loader
    │   ├── model.py                            # Model loading & LoRA setup
    │   └── utils.py                            # Action tokenization utils
    └── rv_eval/                                # Evaluation pipeline
        ├── evaluator.py                        # Episode evaluator
        └── robosuite_env.py                    # Robosuite environment wrapper
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
