import os
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset
import json
from PIL import Image
import sys
sys.path.append("..")
from configs.config import config

class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label, videoID
