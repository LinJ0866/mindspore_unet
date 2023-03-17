import mindspore.nn as nn
import mindspore.ops.operations as P

def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode))
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)

class UnetConv2d(nn.Cell):
    # Convolution block in Unet, usually double conv.
    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_bn = use_bn
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, self.use_bn, self.kernel_size, self.stride, self.padding, 'relu'))
            in_channel = out_channel
        self.convs = nn.SequentialCell(convs)

    def construct(self, input):
        x = self.convs(input)
        return x

class UnetUp(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(UnetUp, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.concat = P.Concat(axis=1)
        self.up_conv = nn.Conv2dTranspose(self.in_channel,
                                 self.out_channel, 
                                 kernel_size=2,
                                 stride=2,
                                 pad_mode="same")
        self.conv = UnetConv2d(in_channel, out_channel, use_bn=False)

    def construct(self, high_feature, *low_feature):
        output = self.up_conv(high_feature)
        for feature in low_feature:
            output = self.concat((output, feature))
        return self.conv(output)

        