import shutil
import torch
import sys
import os
import json
import numpy as np
from configs.config import config
import math
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(save_list, is_best, model, optimizer, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    threshold = save_list[4]
    if(len(config.gpus) > 1):
        state = {
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    filepath = config.model_name + '/' + filename
    # save every model
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, config.best_model + '/' + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def mkdirs():
    if not os.path.exists(config.model_save_path):
        os.mkdir(config.model_save_path)
    if not os.path.exists(config.model_name):
        os.mkdir(config.model_name)
    ##########
    if not os.path.exists(config.best_model):
        os.mkdir(config.best_model)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def adjust_learning_rate(optimizer, epoch, init_param_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= 30):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= 60):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def get_files(path, json_name, num_frames, dataset_name):
    '''
    from every video (frames) to choose num_frames to test
    return: the choosen frames' path and label
    '''

    test_json = json.load(open(path + json_name, 'r'))
    f_choose_json = open(path + 'choose_' + json_name, 'w')
    length = len(test_json)
    # current video need to be preprocessed
    save_photo_name = test_json[0]['photo_path'].split('.')[-3]
    final_json = []

    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = test_json[i]['photo_path']
        photo_label = test_json[i]['photo_label']
        photo_name = photo_path.split('.')[-3]
        if(i == length - 1): # the last one
            photo_frame = int(photo_path.split('.')[-2])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        if(photo_name != save_photo_name or i == length-1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            # filter the first and the last 10 frames
            frame_interval = math.floor((single_video_frame_num - 20) / num_frames)
            for j in range(num_frames):
                dict = {}
                dict['photo_path'] = save_photo_name + '.' + str(single_video_frame_list[10 + j * frame_interval]) + '.png'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                # print(video_number, '###', len(dict))
                final_json.append(dict)
            video_number += 1
            save_photo_name = photo_name
            single_video_frame_list.clear()
            single_video_frame_num = 0

        # get every frame information
        photo_frame = int(photo_path.split('.')[-2])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    print("Total video number: ", video_number, dataset_name)
    json.dump(final_json, f_choose_json, indent=4)
    f_choose_json.close()
