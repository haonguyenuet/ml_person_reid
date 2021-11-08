from torch import nn
from torch.nn import functional as F


# Building Basic Layer
class Conv1x1(nn.Module):
    """ 1x1 Conv -> Batch norm -> ReLU """

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """ 1x1 Conv -> Batch norm """

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """ 1x1 Conv -> DW 3x3 Conv -> Batch norm -> ReLU """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False)
        self.depthwise = nn.Conv2d(
            c_out, c_out, 3, stride=1, padding=1, bias=False, groups=c_out
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvLayer(nn.Module):
    """Conv -> Batch norm -> ReLU"""

    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)

        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Building OS Residual Block


class AggregationGate(nn.Module):

    def __init__(self, c_in):
        super().__init__()
        reduction = 16
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c_in, c_in//reduction,
                             kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c_in//reduction, c_in,
                             kernel_size=1, bias=True, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        output = self.global_avgpool(x)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.activation(output)
        return output * x


class OSResBlock(nn.Module):
    def __init__(self, c_in, c_out, bottleneck_reduction=4, **kwargs):
        super().__init__()
        c_mid = c_out // bottleneck_reduction
        self.conv1 = Conv1x1(c_in, c_mid)
        self.conv2a = LightConv3x3(c_mid, c_mid)
        self.conv2b = nn.Sequential(
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
            LightConv3x3(c_mid, c_mid),
        )
        self.gate = AggregationGate(c_mid)
        self.conv3 = Conv1x1Linear(c_mid, c_out)
        self.downsample = None
        if c_in != c_out:
            self.downsample = Conv1x1Linear(c_in, c_out)

    def forward(self, x):
        residual = self.conv1(x)
        x_a = self.conv2a(residual)
        x_b = self.conv2b(residual)
        x_c = self.conv2c(residual)
        x_d = self.conv2d(residual)
        residual = self.gate(x_a) + self.gate(x_b) + \
            self.gate(x_c) + self.gate(x_d)
        residual = self.conv3(residual)
        if self.downsample is not None:
            x = self.downsample(x)
        out = x + residual
        return F.relu(out)


# Building OSNet for Input 3 x 256 x 128


class OSNet(nn.Module):
    def __init__(self,
                 num_classes=100,
                 blocks=[OSResBlock, OSResBlock, OSResBlock],
                 layers=[2, 2, 2],
                 channels=[64, 256, 384, 512],
                 fc_dim=512):
        super().__init__()
        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.tran1 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.tran2 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer (hidden layer)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True)
        )
        # classifier
        self.classifier = nn.Linear(fc_dim, num_classes)

    def _make_layer(self, block, num_layers, c_in, c_out):
        layers = []

        layers.append(block(c_in, c_out))
        for i in range(1, num_layers):
            layers.append(block(c_out, c_out))

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.tran1(x)
        x = self.conv3(x)
        x = self.tran2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)  # flatten
        v = self.fc(v)
        y = self.classifier(v)
        return y
