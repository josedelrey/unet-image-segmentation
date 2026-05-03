import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, input_size=572, output_size=388):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)

        mask_name = image_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, (self.input_size, self.input_size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.output_size, self.output_size), interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return image, mask