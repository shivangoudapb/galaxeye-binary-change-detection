import os
import numpy as np
import rasterio
from torch.utils.data import Dataset
import albumentations as A


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augment=False):
        self.pre_dir = os.path.join(root_dir, "pre-event")
        self.post_dir = os.path.join(root_dir, "post-event")
        self.mask_dir = os.path.join(root_dir, "target")

        self.files = sorted([
            f for f in os.listdir(self.pre_dir)
            if f.endswith(".tif") or f.endswith(".tiff")
        ])

        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
            ])

    def read_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        return img.astype(np.float32)

    def normalize(self, img):
        img = np.nan_to_num(img)

        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) / (p98 - p2), 0, 1)
        else:
            img = np.zeros_like(img)

        return img

    def remap_mask(self, mask):
        mask = mask.squeeze()

        binary = np.zeros_like(mask, dtype=np.float32)
        binary[(mask == 2) | (mask == 3)] = 1.0

        return binary

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        pre = self.read_tif(os.path.join(self.pre_dir, fname))
        post = self.read_tif(os.path.join(self.post_dir, fname))
        mask = self.read_tif(os.path.join(self.mask_dir, fname))

        pre = self.normalize(pre)
        post = self.normalize(post)
        mask = self.remap_mask(mask)

        merged = np.concatenate([pre, post], axis=2)

        transformed = self.transform(image=merged, mask=mask)

        merged = transformed["image"]
        mask = transformed["mask"]

        merged = np.transpose(merged, (2, 0, 1))  # (C, H, W)

        return (
            merged.astype(np.float32),
            np.expand_dims(mask, 0).astype(np.float32),
        )