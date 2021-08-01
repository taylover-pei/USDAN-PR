from models.backbone import Backbone_resnet
from models.GRL import AdversarialLayer
import torch.nn as nn
import torch
from torch.nn import Parameter

def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class USDAN_model(nn.Module):
    def __init__(self, num_classes):
        super(USDAN_model, self).__init__()
        self.model_backbone = Backbone_resnet()

        self.bottleneck_layer1 = nn.Linear(512, 512)
        self.bottleneck_layer1.weight.data.normal_(0, 0.005)
        self.bottleneck_layer1.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer1,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.kernel = Parameter(torch.Tensor(512, 2))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.classifier_layer = nn.Linear(512, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

        self.add_layer1 = nn.Linear(512, 512)
        self.add_layer2 = nn.Linear(512, 1)
        self.add_layer1.weight.data.normal_(0, 0.01)
        self.add_layer2.weight.data.normal_(0, 0.3)
        self.add_layer1.bias.data.fill_(0.0)
        self.add_layer2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.add_layer1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.add_layer2,
            nn.Sigmoid()
        )

        self.grl_layer = AdversarialLayer(high=1.0)

    def forward(self, x, norm_flag=True):
        feature = self.model_backbone(x)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            # norm feature
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
            # norm weight
            kernel_norm = l2_norm(self.kernel, axis=0)
            classifier_out = torch.mm(feature, kernel_norm)
            adversarial_out = self.ad_net(self.grl_layer(feature))
        else:
            classifier_out = self.classifier_layer(feature)
            adversarial_out = self.ad_net(self.grl_layer(feature))
        return classifier_out, adversarial_out, feature

