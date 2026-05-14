# U-Net Image Segmentation

Work-in-progress PyTorch implementation of a U-Net for binary medical image
segmentation on the ISIC skin lesion dataset.

This repository is currently being developed as an applied computer vision
project. The core training loop, dataset loader, U-Net model, loss function,
augmentation options, threshold tuning, checkpointing, and experiment logging
are in place. The project is still being cleaned up and expanded, so the
README intentionally shows the current state rather than presenting this as a
finished package.

## Current Status

- Implemented a from-scratch U-Net-style encoder-decoder model in PyTorch.
- Uses valid convolutions with `572x572` RGB inputs and `388x388` mask targets,
  following the original U-Net crop-and-concatenate shape pattern.
- Trains on ISIC image/mask pairs with an 80/10/10 train/validation/test split.
- Supports no augmentation, geometric augmentation, mild color augmentation,
  and stronger augmentation.
- Optimizes a combined BCE + Dice loss.
- Tracks validation Dice, saves the best checkpoint, tunes the prediction
  threshold on validation data, then evaluates once on the held-out test split.
- Logs experiment summaries to `results/results.csv`.

## Current Results

The latest local experiment log contains 14 completed runs. The best held-out
test Dice so far is `0.8590`.

| Run | Augmentation | Epochs | LR | Best Val Dice | Test Dice | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `unet_isic_geomaug_lr5e5_e40` | geometric | 40 | `5e-5` | `0.8779` | `0.8590` | `0.5` |
| `unet_isic_mildaug_lr3e5_e40` | mild | 40 | `3e-5` | `0.8762` | `0.8578` | `0.4` |
| `unet_isic_strongaug_lr5e5_e40` | strong | 40 | `5e-5` | `0.8820` | `0.8567` | `0.5` |
| `unet_isic_mildaug_lr7e5_e40` | mild | 40 | `7e-5` | `0.8799` | `0.8546` | `0.5` |
| `unet_isic_mildaug_lr5e5_e40` | mild | 40 | `5e-5` | `0.8775` | `0.8537` | `0.4` |
| `unet_isic_noaug_lr3e5_e40` | none | 40 | `3e-5` | `0.8693` | `0.8485` | `0.4` |
| `unet_isic_noaug_lr7e5_e40` | none | 40 | `7e-5` | `0.8625` | `0.8446` | `0.4` |
| `unet_isic_noaug_lr5e5_e40` | none | 40 | `5e-5` | `0.8611` | `0.8433` | `0.5` |

Early 20-epoch runs reached roughly `0.72-0.73` test Dice for the better
learning-rate settings. Extending training to 40 epochs and adding augmentation
improved the best result to approximately `0.86` test Dice.

## Project Layout

```text
src/
  dataset.py        ISIC dataset loader, preprocessing, and augmentation
  losses.py         Dice loss and BCE + Dice combined loss
  train.py          Training, validation, threshold tuning, and test evaluation
  unet.py           U-Net model definition
  utils.py          Split helpers and Dice metric
  visualize_predictions.py
                    Checkpoint inference and side-by-side prediction panels
run_experiments.py  Batch runner for experiment sweeps
```

Local-only directories are ignored by git:

```text
isic_segmentation/  Dataset images and masks
models/             Saved checkpoints
results/            Experiment CSV logs
```

## Running

This repo is developed in a local Conda environment:

```powershell
conda activate cv
```

Expected Python interpreter:

```text
C:\Users\rlyeh\miniconda3\envs\cv\python.exe
```

Train one experiment:

```powershell
python src/train.py --run_name unet_isic_geomaug_lr5e5_e40 --batch_size 2 --epochs 40 --lr 5e-5 --augmentation_type geomaug
```

Run the current experiment sweep:

```powershell
python run_experiments.py
```

Generate prediction preview panels from a checkpoint:

```powershell
python src/visualize_predictions.py --checkpoint models/unet_isic_geomaug_lr5e5_e40.pth --num_samples 6 --threshold 0.5
```

Panels are written to `results/predictions/` and show:

```text
image | ground truth | prediction
```

The dataset is expected at:

```text
isic_segmentation/images_segmentation/
isic_segmentation/ground_truth/
```

Mask files are expected to follow the pattern:

```text
<image_stem>_segmentation.png
```

## What I Am Working On Next

- Move experiment configuration out of the Python script into a cleaner config
  format.
- Add plots for training curves and validation/test comparisons.
- Add example predictions to the README once the output assets are cleaned and
  ready to commit.
- Improve evaluation reporting beyond Dice, such as IoU and per-image summaries.

## Notes

This is intentionally marked as WIP. The useful parts are already visible in the
code and results, but the repository still needs packaging, documentation, and
visual examples before it should be treated as polished.
