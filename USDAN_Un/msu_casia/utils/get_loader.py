import os
from configs.config import config
import random
import numpy as np
import torch

from utils.utils import get_files
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_loader():
    print('Load Source Data')
    if (config.src_data == 'msu'):
        get_files(config.src_data_label_path, 'train_label.json', config.src_train_num_frames, 'msu')
        src_train_info = open('./data_label/msu_data_label/choose_train_label.json')
        src_train_data_pd = pd.read_json(src_train_info)
    print('Load Taget Data')
    if (config.tgt_data == 'casia'):
        get_files(config.tgt_data_label_path, 'train_label.json', config.tgt_train_num_frames, 'casia')
        tgt_train_info = open('./data_label/casia_data_label/choose_train_label.json')
        tgt_train_data_pd = pd.read_json(tgt_train_info)

    src_train_dataloader = DataLoader(YunpeiDataset(src_train_data_pd, train=True), batch_size=config.batch_size, shuffle=True)
    src_valid_dataloader = DataLoader(YunpeiDataset(src_train_data_pd, train=False), batch_size=1, shuffle=False)
    tgt_train_dataloader = DataLoader(YunpeiDataset(tgt_train_data_pd, train=True), batch_size=config.batch_size, shuffle=True)
    tgt_valid_dataloader = DataLoader(YunpeiDataset(tgt_train_data_pd, train=False), batch_size=1, shuffle=False)

    return src_train_dataloader, src_valid_dataloader, tgt_train_dataloader, tgt_valid_dataloader





