""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
from typing import List

import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable, LoadImage


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
        n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer ** i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

## ---
## changes for randomly picling train data from the training set
## ---

def get_imagse_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    if(mode == 'train'):
        n=6
        label_list = random.sample(label_list, n)
        image_list = [ txt.replace("labels", "images") for txt in label_list ]

        print(label_list)
        print(image_list)

    print(mode, "data length:", len(label_list), len(image_list))
    
    return label_list, image_list


class CryopppDataset(Dataset):
    def __init__(self, args, data_path, name_list, label_list, transform=None, transform_msk=None, mode='train', prompt='random_click',
                 plane=False):

        # label_list, name_list = get_imagse_and_labels_path(data_path, mode)

        self.name_list = name_list
        self.label_list = label_list
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt  # or bboxes
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):

        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        # img_path = os.path.join(self.data_path, self.mode, "images", name)
        img_path = name

        mask_name = self.label_list[index]
        msk_path = mask_name

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')  # ‘L’为灰度图像

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'random_click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        elif self.prompt == 'box':
            pass

        if self.transform:
            state = torch.get_rng_state()  # 返回随机生成器状态(ByteTensor)
            img = self.transform(img)
            torch.set_rng_state(state)  # 设定随机生成器状态

            if self.prompt == 'points_grids':
                point_grids = build_all_layer_point_grids(
                    n_per_side=32,
                    n_layers=0,
                    scale_per_layer=1,
                )
                points_scale = np.array(img.shape[1:])[None, ::-1]
                points_for_image = point_grids[0] * points_scale  # (1024 * 2)
                in_points = torch.as_tensor(points_for_image)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
                # points = (in_points, in_labels)
                pt = points_for_image
                point_label = np.array(in_labels)

            if self.transform_msk:
                mask = self.transform_msk(mask)

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }
