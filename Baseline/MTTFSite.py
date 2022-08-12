import torch.nn as nn
import torch
from typing import Any, Optional, Tuple


class Feature_neT(nn.Module):
    def __init__(self):
        super(Feature_neT, self).__init__()
        self.conv1 = nn.Conv1d(4, 200, 10)
        self.bn_conv1 = nn.BatchNorm1d(200)
        self.conv2 = nn.Conv1d(200, 200, 10)
        self.bn_conv2 = nn.BatchNorm1d(200)
        self.pool1 = nn.AdaptiveAvgPool1d(1)

        self.conv3 = nn.Conv1d(8, 200, 10)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

        self.layernorm1 = nn.LayerNorm(200)
        self.layernorm2 = nn.LayerNorm(200)

        self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten(1)

    def forward(self, s, h):
        s = self.bn_conv1(self.relu(self.conv1(s)))
        s = self.bn_conv2(self.relu(self.conv2(s)))
        s = self.pool1(s)
        s = self.flatten(s)
        s = self.layernorm1(s)

        h = self.relu(self.conv3(h))
        h = self.pool2(h)
        h = self.flatten(h)
        h = self.layernorm2(h)

        y = torch.cat((s, h), 1)
        return y


class Classifier_neT(nn.Module):
    def __init__(self):
        super(Classifier_neT, self).__init__()
        self.cfc1 = nn.Linear(400, 100)
        self.cfc2 = nn.Linear(100, 50)
        self.cfc3 = nn.Linear(50, 1)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cx = self.relu(self.cfc1(x))
        cx = self.relu(self.cfc2(cx))
        cy = self.sigmoid(self.cfc3(cx))

        return cy


class domain_neT(nn.Module):
    def __init__(self, C):
        super(domain_neT, self).__init__()
        self.dfc1 = nn.Linear(400, 100)
        self.dfc2 = nn.Linear(100, 50)
        self.dfc3 = nn.Linear(50, C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        dx = grad_reverse(x)
        dx = self.relu(self.dfc1(dx))
        dx = self.relu(self.dfc2(dx))
        dy = self.dfc3(dx)

        return dy


class GRL(torch.autograd.Function):

    def forward(self: Any, i: torch.Tensor, Lambda: Optional[float] = 1.) -> torch.Tensor:
        self.Lambda = Lambda
        o = Lambda * i
        return o

    def backward(self: Any, grad_o: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_o.neg() * self.Lambda, None


def grad_reverse(x):
    return GRL().apply(x)


class mTTFSite(nn.Module):
    def __init__(self, cell):
        super(mTTFSite, self).__init__()
        self.Feature_neT = Feature_neT()
        self.Classifier_neT = Classifier_neT()
        self.domain_neT = domain_neT(cell)

    def forward(self, s, h, domain=True):
        if domain:
            s_net = self.Feature_neT(s, h)
            d_net = self.domain_neT(s_net)
            return d_net
        else:
            p_net = self.Feature_neT(s, h)
            c_net = self.Classifier_neT(p_net)
            return c_net
