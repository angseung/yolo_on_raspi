from collections import OrderedDict

import torch
import torch.nn as nn
from models.voc.mobilenetv3 import MobileNetV3
from models.voc.yolo_loss import *
from torch.nn import init
from utils.box import nms

import yaml

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class BasicConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, depthwise=False
    ):
        super(BasicConv, self).__init__()
        if depthwise == False:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
                groups=in_channels,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(
        self,
        x,
    ):
        x = self.upsample(x)
        return x


def DepthwiseConvolution(in_filters, out_filters):
    m = nn.Sequential(
        BasicConv(in_filters, in_filters, 3, depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, out_filters, 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, in_filters, 3, depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, filters_list[0], 1),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class Connect(nn.Module):
    def __init__(self, channels):
        super(Connect, self).__init__()

        self.conv = nn.Sequential(
            BasicConv(channels, channels, 3, depthwise=True),
            BasicConv(channels, channels, 1),
        )

    def forward(
        self,
        x,
    ):
        x2 = self.conv(x)
        x = torch.add(x, x2)
        return x


def PartAdd(x, y):
    if x.size(1) == y.size(1):
        return x + y
    len = min(x.size(1), y.size(1))
    new_1 = x[:, :len, ...] + y[:, :len, ...]
    if y.size(1) > x.size(1):
        new_2 = y[:, len:, ...]
    else:
        new_2 = x[:, len:, ...]
    new = torch.cat((new_1, new_2), 1)

    return new


class yolo_graph(nn.Module):
    def __init__(self, config):
        super(yolo_graph, self).__init__()
        self.num_classes = config["yolo"]["num_classes"]
        self.num_anchors = config["yolo"]["num_anchors"]
        #  backbone
        # https://drive.google.com/file/d/1HYPqCM1t8GDj9HnImKitM-QqdR8InxGB/view?usp=sharing
        self.backbone = MobileNetV3("mbv3_large.old.pth.tar")

        self.conv_for_S32 = BasicConv(960, 512, 1)
        # print(num_anchors * (5 + num_classes))
        self.connect_for_S32 = Connect(512)
        self.yolo_headS32 = yolo_head(
            [1024, self.num_anchors * (5 + self.num_classes)], 512
        )

        self.upsample = Upsample(512, 256)
        self.conv_for_S16 = DepthwiseConvolution(160, 256)
        self.connect_for_S16 = Connect(256)
        self.yolo_headS16 = yolo_head(
            [512, self.num_anchors * (5 + self.num_classes)], 256
        )

    def forward(self, x, targets=None):

        feature1, feature2 = self.backbone(x)
        S32 = self.conv_for_S32(feature2)
        S32 = self.connect_for_S32(S32)
        out0 = self.yolo_headS32(S32)
        S32_Upsample = self.upsample(S32)
        S16 = self.conv_for_S16(feature1)
        # S16 = PartAdd(S16,S32_Upsample)
        S16 = torch.add(S16, S32_Upsample)
        S16 = self.connect_for_S16(S16)
        out1 = self.yolo_headS16(S16)

        return out0, out1


# def test():
#    net = yolo(3,20)
#    print(net)

# test()
