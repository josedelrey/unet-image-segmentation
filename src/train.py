import os
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ISICDataset
from src.unet import UNet
from src.losses import BCEDiceLoss
from src.utils import get_image_names, train_test_split, dice_score_from_logits


IMAGE_DIR = "isic_segmentation/images_segmentation"
MASK_DIR = "isic_segmentation/ground_truth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_SIZE = 572
OUTPUT_SIZE = 388

LOG_PATH = "outputs/experiments.csv"

CSV_HEADER = [
    "run_name",
    "augmentation_type",
    "epochs",
    "batch_size",
    "lr",
    "best_epoch",
    "best_val_dice",
    "best_val_loss",
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

    parser.add_argument(
        "--augmentation_type",
        type=str,
        default="noaug",
        choices=["noaug", "lightaug"],
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
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

    return total_loss / len(loader), total_dice / len(loader)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            dice = dice_score_from_logits(logits, masks)

            total_loss += loss.item()
            total_dice += dice

    return total_loss / len(loader), total_dice / len(loader)


def log_experiment(
    run_name,
    augmentation_type,
    epochs,
    batch_size,
    lr,
    best_epoch,
    best_val_dice,
    best_val_loss,
    final_test_dice,
    final_test_loss,
):
    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.exists(LOG_PATH)

    if file_exists:
        with open(LOG_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != CSV_HEADER:
            raise ValueError(
                f"CSV header mismatch in {LOG_PATH}.\n"
                f"Expected: {CSV_HEADER}\n"
                f"Found:    {existing_header}\n"
                "Delete the file or update the header."
            )

    with open(LOG_PATH, "a", newline="") as f:
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
            round(final_test_dice, 4),
            round(final_test_loss, 4),
            "",
        ])


def main():
    args = parse_args()
    set_seed(args.seed)

    model_path = f"models/{args.run_name}.pth"
    os.makedirs("models", exist_ok=True)

    print("Device:", DEVICE)
    print("Run name:", args.run_name)
    print("LR:", args.lr)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Augmentation:", args.augmentation_type)

    image_names = get_image_names(IMAGE_DIR)

    trainval_names, test_names = train_test_split(
        image_names,
        train_ratio=0.8,
        seed=args.seed,
    )

    train_names, val_names = train_test_split(
        trainval_names,
        train_ratio=0.9,
        seed=args.seed,
    )

    print(f"Total images: {len(image_names)}")
    print(f"Train images: {len(train_names)}")
    print(f"Val images:   {len(val_names)}")
    print(f"Test images:  {len(test_names)}")

    train_dataset = ISICDataset(
        IMAGE_DIR,
        MASK_DIR,
        train_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type=args.augmentation_type,
    )

    val_dataset = ISICDataset(
        IMAGE_DIR,
        MASK_DIR,
        val_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type="noaug",
    )

    test_dataset = ISICDataset(
        IMAGE_DIR,
        MASK_DIR,
        test_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        augmentation_type="noaug",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    best_val_loss = None
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Dice: {train_dice:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Best Epoch: {best_epoch}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    final_test_loss, final_test_dice = evaluate(
        model,
        test_loader,
        criterion,
    )

    print(
        f"Final Test Loss: {final_test_loss:.4f} "
        f"Final Test Dice: {final_test_dice:.4f}"
    )

    log_experiment(
        run_name=args.run_name,
        augmentation_type=args.augmentation_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        best_epoch=best_epoch,
        best_val_dice=best_val_dice,
        best_val_loss=best_val_loss,
        final_test_dice=final_test_dice,
        final_test_loss=final_test_loss,
    )


if __name__ == "__main__":
    main()