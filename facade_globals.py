import os
from models import create_model
from util.visualizer import save_images
from util import html

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import numpy as np


class Facade(Dataset):
    def __init__(self, transform, root_dir='./facades/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(os.path.join(self.root_dir, 'image'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        idx = idx + 1
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, f"image/{idx}.jpg")
        label_path = os.path.join(self.root_dir, f"solution/{idx}.jpg")
        
        img = self.transform(Image.open(img_path).convert('RGB'))
        lbl = self.transform(Image.open(label_path).convert('RGB'))

        return img, lbl

data_transform = transforms.Compose([
        # transforms.Resize([256, 256], Image.BICUBIC),
        # transforms.RandomCrop(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
