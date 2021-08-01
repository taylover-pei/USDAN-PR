import torch
import torch.nn as nn

def Center_alignment(source_data, source_label, labeled_target_data, target_label):
    source_size = source_data.size(0)
    target_size = labeled_target_data.size(0)
    source_real = torch.cat([source_data[i].view(1, -1) for i in range(source_size) if (source_label[i].cpu().data.numpy() == 1)])
    source_attack = torch.cat([source_data[i].view(1, -1) for i in range(source_size) if (source_label[i].cpu().data.numpy() == 0)])
    target_real = torch.cat([labeled_target_data[i].view(1, -1) for i in range(target_size) if (target_label[i].cpu().data.numpy() == 1)])
    target_attack = torch.cat([labeled_target_data[i].view(1, -1) for i in range(target_size) if (target_label[i].cpu().data.numpy() == 0)])
    source_real_center = source_real.mean(dim=0).view(1, -1)
    target_real_center = target_real.mean(dim=0).view(1, -1)
    source_attack_center = source_attack.mean(dim=0).view(1, -1)
    target_attack_center = target_attack.mean(dim=0).view(1, -1)
    source_center = torch.cat([source_real_center, source_attack_center], 0)
    target_center = torch.cat([target_real_center, target_attack_center], 0)
    pdist = nn.PairwiseDistance(p=2)
    distance = pdist(source_center, target_center)
    return distance.mean()