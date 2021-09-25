"""
@Author:        ljm
@StartTime:     2021/8/2 20:01
@Filename:      Kervolution.py
"""

# 引入非线性核运算代替普通的卷积运算，希望能提高模型的鲁棒性

import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, x_unf, w, b):
        res = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1,2)
        if b is not None:
            res += b
        return res

class PolynomialKernel(LinearKernel):
    def __init__(self, cp=2.0, dp=3, train_cp=True):
        super(PolynomialKernel, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=train_cp))
        self.dp = dp

    def forward(self, x_unf, w, b):
        ans = (self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp
        return ans

class GaussianKernel(LinearKernel):
    def __init__(self, gamma):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(
            torch.tensor(gamma, requires_grad=True))
    
    def forward(self, x_unf, w, b):
        l = x_unf.transpose(1, 2)[:, :, :, None] - w.view(1, 1, -1, w.size(0))
        l = torch.sum(l**2, 2)
        t =  torch.exp(-self.gamma * l)
        if b is not None:
            t += b
        return t

class KernelConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=partial(GaussianKernel, 0.05),
                stride=1, padding=0, dilation=1, 
                groups=1, bias=None, padding_mode='zeros'):
        super(KernelConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias, padding_mode)
        """
        添加一个卷积核选项， 默认为 PolynomialKernel
        partial(GaussianKernel, 0.05)
        """
        self.kernel_fn = kernel_fn()
    
    # 计算出输出的高和宽
    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding[0] -  1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] -  1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h, w

    def forward(self, x):
        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        h, w = self.compute_shape(x)
        ans = self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)
        return ans


# ResNet 中卷积换成核卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = KernelConv2d(in_planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = KernelConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                KernelConv2d(in_planes, self.expansion*planes, kernel_size=1, 
                                    stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, kernel_fn, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = KernelConv2d(in_planes, planes, kernel_size=1, 
                                bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = KernelConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = KernelConv2d(planes, self.expansion * planes, 
                                kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                KernelConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = KernelConv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Ker_ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def Ker_ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def Ker_ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def Ker_ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def Ker_ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
