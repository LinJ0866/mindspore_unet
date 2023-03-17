import mindspore.nn as nn
import numpy as np

from src.unet.unet_parts import UnetConv2d, UnetUp
from config import cfg

class UNet(nn.Cell):
    def __init__(self, feature_scale=2, use_bn=True):
        super(UNet, self).__init__()
        self.in_channel = 3
        self.feature_scale = feature_scale
        self.use_bn = use_bn

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.conv0 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
        self.conv1 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv2 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv3 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv4 = UnetConv2d(filters[3], filters[4], self.use_bn)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        # Up Sample
        self.up_concat1 = UnetUp(filters[1], filters[0])
        self.up_concat2 = UnetUp(filters[2], filters[1])
        self.up_concat3 = UnetUp(filters[3], filters[2])
        self.up_concat4 = UnetUp(filters[4], filters[3])

        self.final = nn.Conv2d(filters[0], cfg.num_cls, 1)

    def construct(self, input):
        input = input.astype(np.float32)
        x0 = self.conv0(input)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        up4 = self.up_concat4(x4, x3)
        up3 = self.up_concat3(up4, x2)
        up2 = self.up_concat2(up3, x1)
        up1 = self.up_concat1(up2, x0)

        final = self.final(up1)

        return final