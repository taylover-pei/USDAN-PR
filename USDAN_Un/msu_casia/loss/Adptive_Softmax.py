from torch import nn
import torch.nn.functional as F
import torch

class Adaptive_Softmax(nn.Module):

    def __init__(self, focusing_param = 2.0):
        super(Adaptive_Softmax, self).__init__()
        self.focusing_param = focusing_param

    def forward(self, output, target):
        logpt = -F.cross_entropy(output, target)
        pt = torch.exp(logpt)
        adaptive_loss = -((1 - pt) ** self.focusing_param) * logpt
        return adaptive_loss