import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
sys.path.append('..')
from configs.config import config

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = '/home/jiayunpei/uda_dann_anti_spoofing/pretrain_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model

class Backbone_resnet(nn.Module):
    def __init__(self, pretrained=config.pretrained):
        super(Backbone_resnet, self).__init__()
        if (config.model == 'resnet18'):
            model_resnet = resnet18(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        return feature
