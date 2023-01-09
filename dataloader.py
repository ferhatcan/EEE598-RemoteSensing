import os
import json
import torch
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
plt.ion()


class AerialDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.json_file = json_file
        self.transform = transform
        self.root_folder = os.path.split(json_file)[0]

        with open(self.json_file) as f:
            self.data = json.load(f)

        self.cached_hazy_im = [None] * len(self.data)
        self.cached_gt_im = [None] * len(self.data)
        self.cached_t_im = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.cached_gt_im[item] is None:
            hazy_im_path = self.data[str(item)]['train_image']
            gt_im_path = self.data[str(item)]['gt_image']
            transmission_path = self.data[str(item)]['transmission_map']

            hazy_im = Image.open(os.path.join(self.root_folder, hazy_im_path)).convert('RGB')
            gt_im = Image.open(os.path.join(self.root_folder, gt_im_path)).convert('RGB')
            t_im = Image.open(os.path.join(self.root_folder, transmission_path)).convert('RGB')

            # open_cv_image = np.array(hazy_im)[:, :, ::-1].copy()
            # hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV).astype(np.uint8)
            # value = 110 - hsv[:, :, 2].mean()  # whatever value you want to add
            # hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
            # image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)
            # hazy_im = Image.fromarray(image[:, :, ::-1])

            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            if self.transform is not None:
                hazy_im = self.transform(hazy_im)

            random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            if self.transform is not None:
                gt_im = self.transform(gt_im)

            random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            if self.transform is not None:
                t_im = self.transform(t_im)

            self.cached_gt_im[item] = gt_im
            self.cached_hazy_im[item] = hazy_im
            self.cached_t_im[item] = t_im

        else:
            hazy_im, gt_im, t_im = self.cached_hazy_im[item], self.cached_gt_im[item], self.cached_t_im[item]

        return hazy_im, gt_im
        # return hazy_im, t_im


# if __name__ == '__main__':
#     json_file = r'D:\Dataset\GeneratedCloudDataset-Aerial\info.json'
#     train_transforms = torchvision.transforms.Compose([
#         T.Resize(64 + int(.25 * 64)),  # args.img_size + 1/4 *args.img_size
#         T.RandomResizedCrop(64, scale=(0.8, 1.0)),
#         T.ToTensor(),
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     dataset = AerialDataset(json_file, transform=train_transforms)
#     print(len(dataset))
#
#     img, gt = next(iter(dataset))
#
#     for (img, gt) in dataset:
#         print(len(dataset))
#
#     tmp = 'a'
