import os
import random

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ISICDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_names,
        input_size=572,
        output_size=388,
        augmentation_type="noaug",
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.input_size = input_size
        self.output_size = output_size
        self.augmentation_type = augmentation_type

        valid_augmentations = ["noaug", "geomaug", "mildaug", "strongaug"]

        if self.augmentation_type not in valid_augmentations:
            raise ValueError(
                f"Unsupported augmentation_type: {self.augmentation_type}. "
                f"Expected one of: {valid_augmentations}"
            )

    def __len__(self):
        return len(self.image_names)

    def apply_geometric_augmentation(self, image, mask):
        """
        Geometry-only augmentation.

        Applied to both image and mask.
        Use bilinear interpolation for RGB image.
        Use nearest-neighbor interpolation for binary mask.
        """

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        return image, mask

    def apply_mild_augmentation(self, image, mask):
        """
        Mild augmentation.

        Geometry + small brightness/contrast/color changes.
        Color transforms are applied only to the image, not the mask.
        """

        image, mask = self.apply_geometric_augmentation(image, mask)

        if random.random() < 0.3:
            brightness_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.9, 1.1)

            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        if random.random() < 0.2:
            saturation_factor = random.uniform(0.9, 1.1)
            hue_factor = random.uniform(-0.02, 0.02)

            image = TF.adjust_saturation(image, saturation_factor)
            image = TF.adjust_hue(image, hue_factor)

        return image, mask

    def apply_strong_augmentation(self, image, mask):
        """
        Stronger augmentation.

        Geometry + small arbitrary rotation + stronger color perturbation
        + occasional blur.

        This is more aggressive, so it should be tested carefully.
        """

        image, mask = self.apply_geometric_augmentation(image, mask)

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < 0.4:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)

            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        if random.random() < 0.3:
            saturation_factor = random.uniform(0.85, 1.15)
            hue_factor = random.uniform(-0.03, 0.03)

            image = TF.adjust_saturation(image, saturation_factor)
            image = TF.adjust_hue(image, hue_factor)

        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))

        return image, mask

    def apply_augmentation(self, image, mask):
        if self.augmentation_type == "noaug":
            return image, mask

        if self.augmentation_type == "geomaug":
            return self.apply_geometric_augmentation(image, mask)

        if self.augmentation_type == "mildaug":
            return self.apply_mild_augmentation(image, mask)

        if self.augmentation_type == "strongaug":
            return self.apply_strong_augmentation(image, mask)

        raise ValueError(f"Unsupported augmentation_type: {self.augmentation_type}")

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)

        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask to the U-Net input size.
        # The model receives a 572x572 image.
        image = TF.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            (self.input_size, self.input_size),
            interpolation=InterpolationMode.NEAREST,
        )

        # Apply augmentation to both image and mask.
        # For noaug, apply_augmentation should simply return image, mask unchanged.
        image, mask = self.apply_augmentation(image, mask)

        # Original valid-convolution U-Net outputs 388x388 from 572x572.
        # Therefore, the target mask must be the central 388x388 region.
        mask = TF.center_crop(
            mask,
            (self.output_size, self.output_size),
        )

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Convert mask to binary float tensor: background=0, lesion=1.
        mask = (mask > 0.5).float()

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image, mask