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

def retrieve_model():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    
    return model

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train_facade import AdvGAN_Attack #*Using this for changes
from models.networks import define_G

from torch.utils.data import DataLoader, Dataset

import numpy as np
from PIL import Image


import torchvision
import torch
from torchvision import datasets
from torchvision import transforms
import os
import random

import numpy as np

from adv_models.generator import Generator
from adv_models.discriminator import Discriminator

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

def main():
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    #target = define_G(3, 3, 64, 'unet_256', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
    #target.load_state_dict(torch.load('model/gen/gen.mod'))

    model_path = 'facades/latest_net_G.pth'
    dataset_path = 'facades/image/'
    ##Modifications for horse/zebra
    # instantiate the dataset and dataloader
    target = retrieve_model().netG
    #target.load_state_dict(torch.load(model_path))
    #target.eval()
    #target.cuda()
    model = target
    print('Retrieved model')
    dataset = Facade(transform = transforms.ToTensor())  # our custom dataset
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False, drop_last=True)
    advGAN = AdvGAN_Attack("cuda", target, 3)

    #advGAN.train(range(10000), 10)
    print('Running attack')
    advGAN.train(dataloader, 20)

    advGAN.save_results(DataLoader(dataset, batch_size=1))

if __name__ == "__main__":
    main()
