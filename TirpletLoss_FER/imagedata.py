import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os

class ImageData(Dataset):

    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()  ## torch에서 이미지 처리

        if self.is_train:

            self.images = df['pixels']
            self.labels = df['emotion']
            self.index = df.index.values

        else:
            self.images = df['pixels']
            self.labels = df['emotion']
            self.index = df.index.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        imgarray_str = self.images[item].split(' ')
        imgarray = np.asarray(imgarray_str, dtype=np.uint8).reshape(48, 48, 1)
        anchor_img = imgarray

        if self.is_train:
            anchor_label = self.labels[item]

            ## 해당 anchor가 아닌 것들중에서 Label 같은 것들의 index를 가지고 옮
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]

            positive_item = random.choice(positive_list)
            imgarray_str = self.images[positive_item].split(' ')
            imgarray = np.asarray(imgarray_str, dtype=np.uint8).reshape(48, 48, 1)
            positive_img = imgarray

            ## 해당 anchor가 아닌 것들중에서 Label 다른 것들의 index를 가지고 옮
            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]

            nagative_item = random.choice(negative_list)
            imgarray_str = self.images[nagative_item].split(' ')
            imgarray = np.asarray(imgarray_str, dtype=np.uint8).reshape(48, 48, 1)
            negative_img = imgarray

            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img