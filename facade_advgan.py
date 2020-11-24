import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import random
from os import listdir
from os.path import isfile, join
import json
from config import FACE_BOXES_JSON, TEST_DATA_DIR, TRAIN_DATA_DIR

from adv_models.generator import Generator
from adv_models.discriminator import Discriminator

from tqdm import tqdm
from facade_globals import data_transform

# x (unprotectected image) -> y (deep fake)
# x_ = x + perturb (protected) -> y_ (deepfake)
# We want y - y_ to be large

# Input:
# - 1 target video (protect)
# - >5 attack videos (faces are copied from this)
# - set aside 1 attack video for validation
#
# Precompute:
# - Deepfake for each of 5 attack on target video (data/results/train/video_name_{frame}_target_video_{frame}.jpg)
# - Precompute 256x256 face centered    (data/train/video_name/*.jpg)
# - alignments (data/alignments/video_name/*.json)
# - base images (data/base_images/train/*.jpg)
#Folder Structure:
#data
#   - base_images
#         - train
#             -video_name
#                 -image_name
#         - valid
#             -video_name
#                 -image_name
#     - face_centered
#         -train
#             -video_name
#                 -image_name
#         -valid
#             -video_name
#                 -image_name
#     -alignments
#         -video_name
#             -json file mapping common file name to a dict with x,y,w,h of the face alignment box
# #ASSUMPTION: The image name within each subdirectory is standardized and the same throughout




# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """
#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = original_tuple + (path,)

#         last_slash = path.rindex('/')
#         base_path = path[:last_slash]

#         #Obtain alignments - requires the alignment json file located in {dfd,fs}_frames_centered_cropped to be put in path specified in config.py
#         with open(FACE_BOXES_JSON) as f:
#             data = json.load(f)
#         image_name = path.split('/')[-1]
#         x = data[image_name]['x']
#         y = data[image_name]['y']
#         h = data[image_name]['h']
#         w = data[image_name]['w']
#         align_tuple = (x,y,h,w)
#         tuple_with_path_alignment = tuple_with_path + (align_tuple,)

#         ret_tuple = (tuple_with_path_alignment[0], tuple_with_path_alignment[2], tuple_with_path_alignment[3])
#         return ret_tuple


# # instantiate the dataset and dataloader
# train_dataset = ImageFolderWithPaths(
#     root=TRAIN_DATA_DIR, transform=transforms.Compose([transforms.ToTensor()])
# )
# dataloader = DataLoader(train_dataset, batch_size=64, num_workers=1, shuffle=True)

# test_dataset = torchvision.ImageFolderWithPaths(
#     root=TEST_DATA_DIR, transform=torchvision.transforms.ToTensor()
# )

models_path = "./models/instances/"

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.05)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class AdvGAN_Attack:
    def __init__(self, device, target, image_nc, box_min=0, box_max=1):
        output_nc = image_nc
        self.device = device
        self.target = target
        self.input_nc = image_nc
        self.output_nc = output_nc

        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.001)


        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, y):
        """x is the large not cropped face. TODO find a way to associate image with the image it came from (see if we can do it by filename)"""
        # x is the cropped 256x256 to perturb
        x_transformed = x.clone()
        for i in range(x_transformed.shape[0]):
            x_transformed[i] = data_transform(x_transformed[i])


        x = x.cuda()
        y = (self.target(x_transformed) + 1.0) / 2.0
        # optimize D
        perturbation = self.netG(x)

        # add a clipping trick
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)    # 256 x 256

        # apply the adversarial image
        protected_image = adv_images     # TODO: Original image size

        for i in range(1):
            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(y)
            loss_D_real = F.mse_loss(
                pred_real, torch.ones_like(pred_real, device=self.device)
            )
            loss_D_real.backward()

            pred_fake = self.netDisc(protected_image.detach())
            loss_D_fake = F.mse_loss(
                pred_fake, torch.zeros_like(pred_fake, device=self.device)
            )
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(protected_image)
            loss_G_fake = F.mse_loss(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            loss_perturb = torch.mean(
                torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            )

            # 1 - image similarity
            # TODO apply image back to original image
            # ex. perturbed_original = (adv_images patched onto the original image)
            # Clamp it
            # perform face swap with the images

            # Need to see how it affects the

            protected_image_transformed = protected_image.clone()
            for i in range(protected_image_transformed.shape[0]):
                protected_image_transformed[i] = data_transform(protected_image_transformed[i])

            y_ = (self.target(protected_image) + 1) / 2.0
            norm_similarity = torch.abs(torch.dot(torch.flatten(y_ / torch.norm(y_, 2)), torch.flatten(y / torch.norm(y, 2))))
            loss_adv = norm_similarity
            loss_adv.backward(retain_graph=True) # retain graph

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return (
            loss_D_GAN.item(),
            loss_G_fake.item(),
            loss_perturb.item(),
            loss_adv.item(),
        )

    def train(self, train_dataloader, epochs):
        for epoch in tqdm(range(1, epochs + 1)):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
                self.optimizer_D = torch.optim.Adam(
                    self.netDisc.parameters(), lr=0.0001
                )
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.00001)
                self.optimizer_D = torch.optim.Adam(
                    self.netDisc.parameters(), lr=0.00001
                )
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in tqdm(enumerate(train_dataloader, start=0)):
                # noise = torch.randn(64, 100, 1, 1, device=self.device)
                img, lbl = data
                # (images, _, paths) = data
                # images = images.to(self.device)

                (
                    loss_D_batch,
                    loss_G_fake_batch,
                    loss_perturb_batch,
                    loss_adv_batch,
                ) = self.train_batch(lbl, img)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print(
                "epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n"
                % (
                    epoch,
                    loss_D_sum / num_batch,
                    loss_G_fake_sum / num_batch,
                    loss_perturb_sum / num_batch,
                    loss_adv_sum / num_batch,
                )
            )

            # save generator
            if epoch % 5 == 0:
                netG_file_name = models_path + "netG_epoch_" + str(epoch) + ".pth"
                torch.save(self.netG.state_dict(), netG_file_name)

                netD_file_name = models_path + "netD_epoch_" + str(epoch) + ".pth"
                torch.save(self.netDisc.state_dict(), netD_file_name)

    def save_results(self, dataloader):
        for i, data in tqdm(enumerate(dataloader, start=0)):
            img, lbl = data

            x = lbl.cuda()
            y = img.cuda()

            perturbation = self.netG(x)

            # add a clipping trick
            x_ = torch.clamp(perturbation, -0.3, 0.3) + x
            x_ = torch.clamp(x_, self.box_min, self.box_max)    # 256 x 256

            y_ = self.target(data_transform(x_))
            y_true = self.target(data_transform(x))

            x = x.reshape(3, 256, 256)
            y = y.reshape(3, 256, 256)
            perturbation = perturbation.reshape(3, 256, 256)
            x_ = x_.reshape(3, 256, 256)
            y_ = y_.reshape(3, 256, 256)
            y_true = y_true.reshape(3, 256, 256)
            
            y_true = ((y_true + 1) / 2.0 * 255.0).type(torch.ByteTensor)
            y_ = ((y_ + 1) / 2.0 * 255.0).type(torch.ByteTensor)

            save_image(x.cpu(), f"results/x_{i}.jpg")
            save_image(y.cpu(), f"results/y_label_{i}.jpg")
            save_image(perturbation.cpu(), f"results/perturbation_{i}.jpg")
            save_image(x_.cpu(), f"results/protectedX_{i}.jpg")
            save_image(y_.cpu(), f"results/y_hat_{i}.jpg")
            save_image(y_true.cpu(), f"results/y_true_{i}.jpg")
