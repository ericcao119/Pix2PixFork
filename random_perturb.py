import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from tqdm import tqdm


class RandomNoise:

    def __init__(self, target, box_min=-1, box_max=1):
        self.target = target
        self.box_min = box_min
        self.box_max = box_max

    def random_perturb(self, x, max_perturb=0.3, max_norm=0.1):
        """x is the large not cropped face. TODO find a way to associate image with the image it came from (see if we can do it by filename)"""

        # Generate perturbation and restrict the max perturbation to part of a pixel
        perturbation = 2 * torch.rand(1, 3, 256, 256) - 1
        perturbation *= max_perturb
        
        # Restrict the total norm of the vector
        norm = torch.norm(perturbation, 2)
        perturbation *= (max_norm / norm)
        return perturbation

    def save_results(self, dataloader):
        for i, data in tqdm(enumerate(dataloader, start=0)):
            img, lbl = data


            x = lbl.cpu()
            y = img.cpu()
            print(x.shape)
            print(y.shape)

            perturbation = self.random_perturb(x)
    
            # add a clipping trick
            x_ = perturbation + x
            x_ = torch.clamp(x_, self.box_min, self.box_max)    # 256 x 256

            x = x.reshape(3, 256, 256)
            y = y.reshape(3, 256, 256)
            perturbation = perturbation.reshape(3, 256, 256)
            x_ = x_.reshape(3, 256, 256)
            # y_ = y_.reshape(3, 256, 256)
            # y_true = y_true.reshape(3, 256, 256)
            
            # y_true = ((y_true + 1) / 2.0 * 255.0).type(torch.ByteTensor)
            # y_ = ((y_ + 1) / 2.0 * 255.0).type(torch.ByteTensor)

            transforms.ToPILImage()(x.cpu()).save(f"results/random/realLbl_{i}.jpg")
            transforms.ToPILImage()(y.cpu()).save(f"results/random/realImg_{i}.jpg")
            transforms.ToPILImage()(perturbation.cpu()).save(f"results/random/perturbation_{i}.jpg")
            transforms.ToPILImage()(x_.cpu()).save(f"results/random/fakeLbl_{i}.jpg")
            # transforms.ToPILImage()(y_.cpu()).save(f"results/random/fakeImg_{i}.jpg")
            # transforms.ToPILImage()(y_true.cpu()).save(f"results/random/trueOutput_{i}.jpg")