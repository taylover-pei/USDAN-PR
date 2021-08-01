from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate
from utils.evaluate import eval
from utils.get_loader import get_loader
from models.USDAN_model import USDAN_model
from loss.Adptive_Softmax import Adaptive_Softmax
from loss.Center_alignment import Center_alignment
from loss.Entropy import EntropyLoss
import random
import numpy as np
from configs.config import config
from datetime import datetime
import time

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():
    mkdirs()
    src_train_dataloader, src_valid_dataloader, \
    tgt_train_dataloader, tgt_valid_dataloader, labeled_train_dataloader = get_loader()

    best_model_ACC = 0.0
    best_model_EER = 1.0
    best_model_HTER = 1.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:threshold
    valid_args = [np.inf, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()
    loss_adversarial = AverageMeter()

    net = USDAN_model(config.num_classes)
    net = net.to(device)

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_train_USDAN_Semi.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('** start training target model! **\n')
    log.write(
        '--------|---------- VALID ---------|--- classifier ---|-- adversarial --|-- Current Best --|\n')
    log.write(
        '  iter  |   loss    top-1   HTER   |   loss   top-1   |      loss       |  top-1    HTER   |\n')
    log.write(
        '-------------------------------------------------------------------------------------------|\n')

    criterion = {
        'adaptive_softmax': Adaptive_Softmax().cuda(),
        'softmax': nn.CrossEntropyLoss().cuda(),
        'adversarial': nn.BCELoss().cuda()
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.model_backbone.parameters()), "lr": 0.01},
        {"params": filter(lambda p: p.requires_grad, net.bottleneck_layer.parameters()), "lr": 0.01},
        {"params": filter(lambda p: p.requires_grad, net.classifier_layer.parameters()), "lr": 0.01},
        {"params": filter(lambda p: p.requires_grad, net.ad_net.parameters()), "lr": 0.01}
    ]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=5e-4)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    tgt_train_iter = iter(tgt_train_dataloader)
    src_train_iter = iter(src_train_dataloader)
    labeled_tgt_train_iter = iter(labeled_train_dataloader)
    tgt_iter_per_epoch = len(tgt_train_iter)
    src_iter_per_epoch = len(src_train_iter)
    labeled_tgt_iter_per_epoch = len(labeled_tgt_train_iter)
    iter_per_epoch = min(tgt_iter_per_epoch, src_iter_per_epoch)

    max_iter = 2000
    epoch = 1
    if (config.resume == True):
        checkpoint = torch.load(config.best_model + config.tgt_best_model_name)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint['epoch']
        print('\n**epoch: ',epoch)

    for iter_num in range(max_iter+1):

        if(iter_num % src_iter_per_epoch == 0):
            src_train_iter = iter(src_train_dataloader)
        if(iter_num % tgt_iter_per_epoch == 0):
            tgt_train_iter = iter(tgt_train_dataloader)
        if (iter_num % labeled_tgt_iter_per_epoch == 0):
            labeled_tgt_train_iter = iter(labeled_train_dataloader)
        if(iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr)

        input_src, source_label = src_train_iter.next()
        input_tgt, target_label = tgt_train_iter.next()
        labeled_input_tgt, labeled_target_label = labeled_tgt_train_iter.next()
        input = torch.cat((input_src, input_tgt), dim=0)
        input = torch.cat((input, labeled_input_tgt), dim=0)
        adversarial_label = torch.from_numpy(np.array([[1], ] * input_src.size(0) + [[0],] * input_tgt.size(0))).float()
        input = input.to(device)
        source_label = source_label.to(device)
        target_label = target_label.to(device)
        adversarial_label = adversarial_label.to(device)
        labeled_target_label = labeled_target_label.to(device)
        classifier_out, adversarial_out, feature = net(input, norm_flag=config.norm_flag)

        cls_loss = criterion["adaptive_softmax"](classifier_out.narrow(0, 0, input_src.size(0)), source_label)
        labeled_tgt_cls_loss = criterion["adaptive_softmax"](
            classifier_out.narrow(0, input_src.size(0)+input_tgt.size(0), labeled_input_tgt.size(0)),
            labeled_target_label)
        softmax_out = nn.Softmax(dim=1)(classifier_out.narrow(0, input_src.size(0), input_tgt.size(0)))
        entropy_loss = EntropyLoss(softmax_out)
        transfer_loss = criterion["adversarial"](adversarial_out.narrow(0, 0, input_src.size(0)+input_tgt.size(0)), adversarial_label)
        alignment_loss = Center_alignment(feature.narrow(0, 0, input_src.size(0)), source_label,
                                   feature.narrow(0, input_src.size(0)+input_tgt.size(0), labeled_input_tgt.size(0)), labeled_target_label)

        total_loss = 7.5 * cls_loss + \
                     0.1 * entropy_loss + \
                     2 * transfer_loss + \
                     7.5 * labeled_tgt_cls_loss + \
                     1 * alignment_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        loss_classifier.update(cls_loss.item())
        loss_adversarial.update(transfer_loss.item())
        acc = accuracy(classifier_out.narrow(0, 0, input_src.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |     %6.3f      |  %6.3f  %6.3f  |'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[3] * 100,
                loss_classifier.avg, classifer_top1.avg,
                loss_adversarial.avg,
                float(best_model_ACC), float(best_model_HTER * 100))
            , end='', flush=True)

        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            valid_args = eval(tgt_valid_dataloader, net, epoch)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[4]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[1]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, threshold]
            save_checkpoint(save_list, is_best, net, optimizer)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |     %6.3f      |  %6.3f  %6.3f  |'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[3] * 100,
                loss_classifier.avg, classifer_top1.avg,
                loss_adversarial.avg,
                float(best_model_ACC), float(best_model_HTER * 100)))
            log.write('\n')
            time.sleep(0.01)

if __name__ == '__main__':
    train() 










