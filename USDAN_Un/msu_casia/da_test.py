
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from configs.config import config
from utils.utils import get_files
from utils.dataset import YunpeiDataset
from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr
from models.USDAN_model import USDAN_model

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

def test(test_dataloader, model, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
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
                number += 1
                if (number % 100 == 0):
                    print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
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
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_top1.update(acc_valid[0])
    cur_EER_valid, _, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, threshold]

def main():
    net = USDAN_model(config.num_classes).to('cuda')
    # load dataset
    get_files(config.test_label_path, 'test_label.json', config.test_num_frames, config.test_data)
    test_info = open(config.test_label_path + 'choose_test_label.json')
    test_data_pd = pd.read_json(test_info)
    test_dataloader = DataLoader(YunpeiDataset(test_data_pd, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    # load model
    net_ = torch.load(config.best_model + config.tgt_best_model_name)
    net.load_state_dict(net_["state_dict"])
    threshold = net_["threshold"]
    # test model
    test_args = test(test_dataloader, net, threshold)
    print('\n===========Test Info===========\n')
    print('threshold: ', threshold)
    print(config.test_data, 'Test acc: %5.3f' %(test_args[0]))
    print(config.test_data, 'Test EER: %5.3f' %(test_args[1]))
    print(config.test_data, 'Test HTER: %5.3f' %(test_args[2]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()
