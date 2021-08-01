from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr
from torch.autograd import Variable
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from configs.config import config

def eval(valid_dataloader, model, epoch):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            classifier_out, _, _ = model(input, norm_flag=config.norm_flag)
            prob = F.softmax(classifier_out, dim=1)
            prob = prob.cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()[0]
            if(videoID in prob_dict.keys()):
                prob_dict[videoID].append(prob[0])
                label_dict[videoID].append(label[0])
                output_dict_tmp[videoID].append(classifier_out)
                target_dict_tmp[videoID].append(target)
            else:
                prob_dict[videoID] = []
                label_dict[videoID] = []
                prob_dict[videoID].append(prob[0])
                label_dict[videoID].append(label[0])
                output_dict_tmp[videoID] = []
                target_dict_tmp[videoID] = []
                output_dict_tmp[videoID].append(classifier_out)
                target_dict_tmp[videoID].append(target)
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target)
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, threshold]

