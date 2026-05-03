import os
import torch
from torch.utils.data import DataLoader

from src.dataset import ISICDataset
from src.unet import UNetOriginal
from src.losses import BCEDiceLoss
from src.utils import get_image_names, train_test_split, dice_score_from_logits


IMAGE_DIR = "isic_segmentation/images_segmentation"
MASK_DIR = "isic_segmentation/ground_truth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4

INPUT_SIZE = 572
OUTPUT_SIZE = 388


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


def main():
    image_names = get_image_names(IMAGE_DIR)
    train_names, test_names = train_test_split(image_names, train_ratio=0.8)

    train_dataset = ISICDataset(
        IMAGE_DIR,
        MASK_DIR,
        train_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    test_dataset = ISICDataset(
        IMAGE_DIR,
        MASK_DIR,
        test_names,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = UNetOriginal(in_channels=3, out_channels=1).to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_dice = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        test_loss, test_dice = evaluate(
            model, test_loader, criterion
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Dice: {train_dice:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Dice: {test_dice:.4f}"
        )

        if test_dice > best_test_dice:
            best_test_dice = test_dice
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/unet_original_isic.pth")

    print(f"Best test Dice: {best_test_dice:.4f}")


if __name__ == "__main__":
    main()