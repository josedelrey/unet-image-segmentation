import os
import random
from PIL import Image
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

    def __len__(self):
        return len(self.image_names)

    def apply_light_augmentation(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

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

        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)

            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)

        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            (self.output_size, self.output_size),
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augmentation_type == "lightaug":
            image, mask = self.apply_light_augmentation(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image, mask