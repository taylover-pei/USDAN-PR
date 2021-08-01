import os
import random
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
from configs.config import config
from utils.dataset import YunpeiDataset
from utils.utils import get_files, target_data_sample

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_loader():
    print('Load Source Data (MSU)')
    get_files(config.src_data_label_path, 'train_label.json', config.src_train_num_frames, 'msu')
    src_train_info = open('./data_label/msu_data_label/choose_train_label.json')
    src_train_data_pd = pd.read_json(src_train_info)

    print('Load Taget Data (CASIA)')
    get_files(config.tgt_data_label_path, 'train_label.json', config.tgt_train_num_frames, 'casia')
    tgt_train_info = open('./data_label/casia_data_label/choose_train_label.json')
    tgt_train_data_pd = pd.read_json(tgt_train_info)
    # labeled target data sample
    target_data_sample(config.tgt_data_label_path, 'choose_train_label.json', labeled_video_num=3,
                       data_pd=tgt_train_data_pd, num_frame_per_video=config.tgt_train_num_frames)
    labeled_tgt_train_info = open('./data_label/casia_data_label/labeled_choose_train_label.json')
    labeled_tgt_train_data_list = pd.read_json(labeled_tgt_train_info)

    src_train_dataloader = DataLoader(YunpeiDataset(src_train_data_pd, train=True), batch_size=config.batch_size, shuffle=True)
    # just use the training set as the validation set
    src_valid_dataloader = DataLoader(YunpeiDataset(src_train_data_pd, train=False), batch_size=1, shuffle=False)
    tgt_train_dataloader = DataLoader(YunpeiDataset(tgt_train_data_pd, train=True), batch_size=config.batch_size, shuffle=True)
    # just use the training set as the validation set
    tgt_valid_dataloader = DataLoader(YunpeiDataset(tgt_train_data_pd, train=False), batch_size=1, shuffle=False)
    labeled_tgt_train_dataloader = DataLoader(YunpeiDataset(labeled_tgt_train_data_list, train=True), batch_size=config.batch_size, shuffle=True)

    return src_train_dataloader, src_valid_dataloader, tgt_train_dataloader, tgt_valid_dataloader, labeled_tgt_train_dataloader





