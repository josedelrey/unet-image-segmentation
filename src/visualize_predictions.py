import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from dataset import ISICDataset
from unet import UNet
from utils import get_image_names, train_test_split


IMAGE_DIR = "isic_segmentation/images_segmentation"
MASK_DIR = "isic_segmentation/ground_truth"

INPUT_SIZE = 572
OUTPUT_SIZE = 388

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side image, ground-truth, and prediction panels."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/unet_isic_geomaug_lr5e5_e40.pth",
        help="Path to a trained U-Net checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/predictions",
        help="Directory where visualization panels will be written.",
    )
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--mask_dir", type=str, default=MASK_DIR)

    return parser.parse_args()


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def get_split_names(image_names, split, seed):
    trainval_names, test_names = train_test_split(
        image_names,
        train_ratio=0.8,
        seed=seed,
    )
    train_names, val_names = train_test_split(
        trainval_names,
        train_ratio=0.9,
        seed=seed,
    )

    if split == "train":
        return train_names
    if split == "val":
        return val_names
    return test_names


def tensor_image_to_pil(image_tensor):
    image = (image_tensor.cpu() * STD + MEAN).clamp(0.0, 1.0)
    image = TF.center_crop(image, (OUTPUT_SIZE, OUTPUT_SIZE))
    return TF.to_pil_image(image)


def mask_tensor_to_pil(mask_tensor):
    mask = mask_tensor.squeeze().detach().cpu().clamp(0.0, 1.0)
    return TF.to_pil_image(mask)


def binary_dice(pred_mask, target_mask, eps=1e-7):
    pred = pred_mask.float().view(-1)
    target = target_mask.float().view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)
    return dice.item()


def make_labeled_panel(image, gt_mask, pred_mask, image_name, dice):
    header_height = 32
    footer_height = 20
    padding = 8
    panel_width = OUTPUT_SIZE * 3 + padding * 4
    panel_height = OUTPUT_SIZE + header_height + footer_height + padding * 3

    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    columns = [
        ("Image", image.convert("RGB")),
        ("Ground truth", gt_mask.convert("RGB")),
        (f"Prediction | Dice {dice:.3f}", pred_mask.convert("RGB")),
    ]

    for col_idx, (label, item) in enumerate(columns):
        x = padding + col_idx * (OUTPUT_SIZE + padding)
        draw.text((x, padding), label, fill=(20, 20, 20), font=font)
        panel.paste(item, (x, header_height + padding))

    draw.text(
        (padding, header_height + OUTPUT_SIZE + padding * 2),
        image_name,
        fill=(80, 80, 80),
        font=font,
    )

    return panel


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_names = get_image_names(args.image_dir)
    split_names = get_split_names(image_names, args.split, args.seed)
    selected_names = split_names[:args.num_samples]

    dataset = ISICDataset(
        args.image_dir,
        args.mask_dir,
        selected_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type="noaug",
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Threshold: {args.threshold}")
    print(f"Writing panels to: {output_dir}")

    with torch.no_grad():
        for idx, image_name in enumerate(selected_names):
            image, gt_mask = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            pred_mask = (torch.sigmoid(logits).cpu().squeeze(0) > args.threshold).float()

            dice = binary_dice(pred_mask, gt_mask)

            panel = make_labeled_panel(
                tensor_image_to_pil(image),
                mask_tensor_to_pil(gt_mask),
                mask_tensor_to_pil(pred_mask),
                image_name=image_name,
                dice=dice,
            )

            stem = Path(image_name).stem
            output_path = output_dir / f"{idx + 1:02d}_{stem}_prediction.png"
            panel.save(output_path)
            print(f"Saved {output_path} | Dice: {dice:.4f}")


if __name__ == "__main__":
    main()
