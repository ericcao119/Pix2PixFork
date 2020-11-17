"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

from util.util import tensor2im

from random_perturb import RandomNoise
from train_facade import AdvGAN_Attack #*Using this for changes

class Facade(Dataset):
    def __init__(self, root_dir='./facades/', transform=None):
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

if __name__ == '__main__':
    gen = RandomNoise(None)

    dataset = Facade(transform = transforms.ToTensor())  # our custom dataset
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=True)

    gen.save_results(dataloader)