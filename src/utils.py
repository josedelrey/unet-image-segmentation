import os
import random
import torch
import numpy as np


def get_image_names(image_dir):
    valid_exts = [".jpg", ".jpeg", ".png"]
    return [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]


def train_test_split(names, train_ratio=0.8, seed=42):
    random.seed(seed)
    names = names.copy()
    random.shuffle(names)

    split_idx = int(len(names) * train_ratio)
    return names[:split_idx], names[split_idx:]


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + eps) / (
        preds.sum() + targets.sum() + eps
    )

    return dice.item()