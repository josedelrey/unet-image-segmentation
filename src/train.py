import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ISICDataset
from unet import UNet
from losses import BCEDiceLoss
from utils import get_image_names, train_test_split, dice_score_from_logits


DEFAULT_IMAGE_DIR = "isic_segmentation/images_segmentation"
DEFAULT_MASK_DIR = "isic_segmentation/ground_truth"
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_PATH = "results/results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_SIZE = 572
OUTPUT_SIZE = 388

CSV_HEADER = [
    "run_name",
    "augmentation_type",
    "epochs",
    "batch_size",
    "lr",
    "best_epoch",
    "best_val_dice",
    "best_val_loss",
    "best_threshold",
    "final_test_dice",
    "final_test_loss",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results_path", type=str, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--augmentation_type",
        type=str,
        default="noaug",
        choices=["noaug", "geomaug", "mildaug", "strongaug"],
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()

    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", leave=False)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = dice_score_from_logits(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1):.4f}",
            "dice": f"{total_dice / (batch_idx + 1):.4f}",
        })

    return total_loss / len(loader), total_dice / len(loader)


def evaluate(model, loader, criterion, desc="Val", threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            dice = dice_score_from_logits(logits, masks, threshold=threshold)

            total_loss += loss.item()
            total_dice += dice

            pbar.set_postfix({
                "loss": f"{total_loss / (batch_idx + 1):.4f}",
                "dice": f"{total_dice / (batch_idx + 1):.4f}",
            })

    return total_loss / len(loader), total_dice / len(loader)


def find_best_threshold(model, loader, thresholds):
    best_threshold = 0.5
    best_dice = 0.0

    print("\nTuning threshold on validation set...")

    for threshold in thresholds:
        _, val_dice = evaluate(
            model,
            loader,
            criterion=torch.nn.BCEWithLogitsLoss(),
            desc=f"Threshold {threshold:.2f}",
            threshold=threshold,
        )

        print(f"Threshold {threshold:.2f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_threshold = threshold

    return best_threshold, best_dice


def log_experiment(
    results_path,
    run_name,
    augmentation_type,
    epochs,
    batch_size,
    lr,
    best_epoch,
    best_val_dice,
    best_val_loss,
    best_threshold,
    final_test_dice,
    final_test_loss,
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = results_path.exists()

    if file_exists:
        with results_path.open("r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != CSV_HEADER:
            raise ValueError(
                f"CSV header mismatch in {results_path}.\n"
                f"Expected: {CSV_HEADER}\n"
                f"Found:    {existing_header}\n"
                "Delete the file or update the header."
            )

    with results_path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            run_name,
            augmentation_type,
            epochs,
            batch_size,
            lr,
            best_epoch,
            round(best_val_dice, 4),
            round(best_val_loss, 4),
            round(best_threshold, 2),
            round(final_test_dice, 4),
            round(final_test_loss, 4),
            "",
        ])


def main():
    args = parse_args()
    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.run_name}.pth"

    print("Device:", DEVICE)
    print("Run name:", args.run_name)
    print("LR:", args.lr)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Augmentation:", args.augmentation_type)
    print("Image dir:", args.image_dir)
    print("Mask dir:", args.mask_dir)
    print("Model path:", model_path)
    print("Results path:", args.results_path)
    print("DataLoader workers:", args.num_workers)

    image_names = get_image_names(args.image_dir)

    trainval_names, test_names = train_test_split(image_names, train_ratio=0.8, seed=args.seed)
    train_names, val_names = train_test_split(trainval_names, train_ratio=0.9, seed=args.seed)

    print(f"Total images: {len(image_names)}")
    print(f"Train images: {len(train_names)}")
    print(f"Val images:   {len(val_names)}")
    print(f"Test images:  {len(test_names)}")

    train_dataset = ISICDataset(
        args.image_dir,
        args.mask_dir,
        train_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type=args.augmentation_type,
    )

    val_dataset = ISICDataset(
        args.image_dir,
        args.mask_dir,
        val_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type="noaug",
    )

    test_dataset = ISICDataset(
        args.image_dir,
        args.mask_dir,
        test_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type="noaug",
    )

    pin_memory = DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_dice = 0.0
    best_val_loss = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            args.epochs,
        )

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            criterion,
            desc=f"Val   [{epoch}/{args.epochs}]",
            threshold=0.5,
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {current_lr:.6f} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Dice: {train_dice:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    print(f"Best Val Dice at threshold 0.5: {best_val_dice:.4f}")
    print(f"Best Epoch: {best_epoch}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_threshold, best_threshold_val_dice = find_best_threshold(
        model,
        val_loader,
        thresholds,
    )

    print(
        f"Best threshold: {best_threshold:.2f} "
        f"with Val Dice: {best_threshold_val_dice:.4f}"
    )

    final_test_loss, final_test_dice = evaluate(
        model,
        test_loader,
        criterion,
        desc="Final Test",
        threshold=best_threshold,
    )

    print(
        f"Final Test Loss: {final_test_loss:.4f} "
        f"Final Test Dice: {final_test_dice:.4f} "
        f"Threshold: {best_threshold:.2f}"
    )

    log_experiment(
        results_path=args.results_path,
        run_name=args.run_name,
        augmentation_type=args.augmentation_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        best_epoch=best_epoch,
        best_val_dice=best_threshold_val_dice,
        best_val_loss=best_val_loss,
        best_threshold=best_threshold,
        final_test_dice=final_test_dice,
        final_test_loss=final_test_loss,
    )


if __name__ == "__main__":
    main()
