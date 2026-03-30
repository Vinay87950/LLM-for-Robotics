# Migration Notes (Original Code) this .md file is not modified

Technical details on differences between this repo and the [original VLA-0](https://github.com/NVlabs/vla0).

---

## Quick Reference

### Train

| Feature | Original | This Repo | Status |
|---------|----------|-----------|--------|
| Multi-GPU | `mp.spawn` + DDP | `accelerate` | ✅ |
| AMP | `cfg.EXP.AMP` | `SFTConfig.bf16` | ✅ |
| Gradient checkpointing | `cfg.TRAIN.grad_checkpoint` | `SFTConfig.gradient_checkpointing` | ✅ |
| Resume checkpoint | `--resume` | `SFTConfig.resume_from_checkpoint` | ✅ |
| WandB | ❌ | `SFTConfig.report_to="wandb"` | ✅ New |

### Eval

| Feature | Original | This Repo | Status |
|---------|----------|-----------|--------|
| `action_horizon` | 0 → fallback to 8 | 1 | ⚠️ Different |
| `ensemble_prediction` | v1 & v2 | v1 & v2 | ✅ |
| `skip_evaluated` | ✅ | ✅ | ✅ |
| `torch_compile` | `--no-torch-compile` | `--torch_compile` | ✅ |

---

## 1. Training Configuration

### 1.1 DDP and Epochs

The original does not use `DistributedSampler`. Each GPU sees the full dataset per epoch.

| | Original | This Repo |
|--|----------|-----------|
| Sampler | None (shuffle=True) | DistributedSampler |
| Data per GPU per epoch | 100% | 100% / N GPUs |
| Epochs to match | 24 | 24 × 8 = 192 |

**Note**: The author confirmed they stopped training early and didn't complete all 24 epochs. We use **32 epochs** which is sufficient for convergence.

### 1.2 Learning Rate

Original scales LR by GPU count:

```python
optimizer = AdamW(params, lr=cfg.TRAIN.lr * num_gpus)  # 5e-6 × 8 = 4e-5
```

This repo uses 4e-5 directly. Without this, training does not converge.

### 1.3 Summary

| Parameter | Original | This Repo |
|-----------|----------|-----------|
| LR | 5e-6 × 8 = 4e-5 | 4e-5 |
| Epochs | 24 (early stopped) | 32 |
| Batch (per GPU) | 8 | 8 |
| Gradient clip | 0.0 | 0.0 |

---

## 2. Eval Behavior

### 2.1 action_horizon

```
horizon = 8        → Model predicts 8 actions
action_horizon = N → Execute N, then re-query
```

| | Default | Behavior |
|--|---------|----------|
| Original | 0 | Falls back to config horizon = 8 |
| This Repo | 1 | Re-query every step |

Use `--action_horizon 8` to match original.

### 2.2 Ensemble Prediction

Averages overlapping action chunks.

```
chunks:  [A,   B,   C  ]  (oldest → newest)
weights: [0.5, 0.5, 1.0]
result = (0.5A + 0.5B + C) / 2 = 0.25A + 0.25B + 0.5C
```

Version 1: All old chunks get flat 0.5 weight.
Version 2: Exponential decay (`weight^(n-i-1)`).

---

## 3. Verified Equivalence

Run: `pytest scripts/verify_migration.py -v`

| Component | Status |
|-----------|--------|
| Action Discretization | ✅ Match |
| Image Tiling | ✅ Match |
| Label Masking | ✅ Match |
| Collator Output | ✅ Match |

### Collator Output

| Metric | Value |
|--------|-------|
| Total tokens | 433 |
| Masked (labels=-100) | 226 |
| Unmasked | 207 |
| Vision tokens | 128 |
| Action values | 56 (8×7) |

**Decoded**:
```
<|im_start|>system
Analyze the input image and predict robot actions for the next 8 timesteps. Each action has 7 dimensions. Output a single sequence of 56 integers (0-1000 each), representing the 8 timesteps sequentially. Provide only space separated numbers. Nothing else.<|im_end|>
<|im_start|>user
Picture 1: <|vision_start|><|image_pad|>...(128 pads)...<|vision_end|>put the white mug on the left plate and put the yellow and white mug on the right plate<|im_end|>
<|im_start|>assistant
474 479 460 669 391 674 0 471 479 460 669 391 674 0 458 479 460 669 391 674 0 455 472 460 669 391 674 0 455 444 460 669 391 674 0 455 396 449 669 391 674 0 449 335 422 669 391 674 0 422 282 397 669 391 674 0<|im_end|>
```
**My Output**:
```
<|im_start|>system
Analyze the input image and predict robot actions for the next 8 timesteps. Each action has 7 dimensions. Output a single sequence of 56 integers (0-1000 each), representing the 8 timesteps sequentially. Provide only space separated numbers. Nothing else.<|im_end|>
<|im_start|>user
Picture 1: <|vision_start|><|image_pad|>...(128 pads)...<|vision_end|>lift the cube from the table<|im_end|>
<|im_start|>assistant
250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0 250 250 250 209 304 357 0<|im_end|>
```


---

## 4. File Mapping

| Original | This Repo |
|----------|-----------|
| `rv_train/train.py` | `scripts/train.py` |
| `rv_train/models/qwen/model.py` | `src/rv_train/model.py` |
| `rv_train/data/libero_dataset.py` | `src/rv_train/dataset.py` |
| `libs/RoboVerse/roboverse/evals/` | `src/rv_eval/` |
| `eval/eval_libero.py` | `scripts/eval.py` |

---

## 5. Known Differences

- **Stats**: Original uses pre-computed; this repo reads from LeRobotDataset metadata
- **Epochs**: Original claims 24 but was early-stopped; this repo uses 32
