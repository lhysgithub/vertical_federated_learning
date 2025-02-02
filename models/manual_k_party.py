import torch.nn as nn
from . import resnet as models
# from . import resnet_for_cifar as models
import torch


class Manual_A(nn.Module):

    def __init__(self, num_classes, layers, in_channel=3, u_dim=64, k=2, width=1):
        super(Manual_A, self).__init__()
        if layers == 18:
            self.net = models.resnet18(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 34:
            self.net = models.resnet34(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 50:
            self.net = models.resnet50(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 101:
            self.net = models.resnet101(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 19:
            self.net = models.mobilenet_v2(pretrained=False, num_classes=u_dim)
        elif layers == 51:
            self.net = models.shufflenet_v2_x1_0(pretrained=False, num_classes=u_dim)
        elif layers == 52:
            self.net = models.squeezenet1_0(pretrained=False, num_classes=u_dim)
        else:
            raise ValueError("Wrong number of layers for model")
        self.classifier = nn.Linear(u_dim * k, num_classes)

    def forward(self, input, U_B):
        out = self.net(input)
        if U_B is not None:
            out = torch.cat([out] + [U for U in U_B], dim=1)
        logits = self.classifier(out)
        return logits


class Manual_B(nn.Module):

    def __init__(self, layers, in_channel=3, u_dim=64, width=1):
        super(Manual_B, self).__init__()
        if layers == 18:
            self.net = models.resnet18(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 34:
            self.net = models.resnet34(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 50:
            self.net = models.resnet50(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 101:
            self.net = models.resnet101(pretrained=False, low_dim=u_dim, in_channel=in_channel, width=width)
        elif layers == 19:
            self.net = models.mobilenet_v2(pretrained=False, num_classes=u_dim)
        elif layers == 51:
            self.net = models.shufflenet_v2_x1_0(pretrained=False, num_classes=u_dim)
        elif layers == 52:
            self.net = models.squeezenet1_0(pretrained=False, num_classes=u_dim)
        else:
            raise ValueError("Wrong number of layers for model")

    def forward(self, input):
        out = self.net(input)
        return out