from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.gt_folder = os.path.join(root_dir, 'GT')
        self.original_folder = os.path.join(root_dir, 'input')
        self.images = os.listdir(self.gt_folder)

        # Split the dataset into train, validation, and test sets
        train_images, test_images = train_test_split(self.images, test_size=0.2, random_state=42)
        val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

        if split == 'train':
            self.images = train_images
        elif split == 'val':
            self.images = val_images
        elif split == 'test':
            self.images = test_images
        else:
            raise ValueError("Invalid split type. Use 'train', 'val', or 'test'.")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        gt_img_path = os.path.join(self.gt_folder, img_name)
        original_img_path = os.path.join(self.original_folder, img_name)

        gt_img = Image.open(gt_img_path).convert("RGB")
        original_img = Image.open(original_img_path).convert("RGB")

        gt_img = self.transform(gt_img)
        original_img = self.transform(original_img)

        return gt_img, original_img