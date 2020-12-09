from torch import nn
import torch
import numpy as np
from utils.config import get_config
config = get_config()
r1 = config.protocol.r1
TR = config.protocol.TR
alpha = config.protocol.alpha/180*np.pi
deltt = config.protocol.deltt/60
class CNNdFeature(nn.Module):
    def __init__(self):
        super(CNNdFeature, self).__init__()
        self.feature_extra = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 56
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 28
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 14
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # 7
            nn.ReLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 56
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),  # 28
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=8, dilation=8),  # 14
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=25, dilation=25),  # 7
            nn.ReLU(),
        )


    def forward(self, signal, T10):
        extra_data = torch.ones_like(signal[:,0:1,...])
        extra_data[:, 0, ...] = extra_data[:, 0, ...]*T10
        # extra_data[:, 1, ...] = extra_data[:, 1, ...]*T1b0
        in_data = torch.cat([signal, extra_data], dim=1)
        cnn_feature = self.feature_extra(in_data)
        local_feature = self.local(cnn_feature)
        wide_feature = self.wide(cnn_feature)
        return torch.cat([local_feature, wide_feature], dim=1)
class CNNFeature(nn.Module):
    def __init__(self):
        super(CNNFeature, self).__init__()
        self.feature_extra = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
        )
    def forward(self, signal, T10):
        extra_data = torch.ones_like(signal[:, 0:1, ...])
        extra_data[:, 0, ...] = extra_data[:, 0, ...]*T10
        in_data = torch.cat([signal, extra_data], dim=1)
        cnn_feature = self.feature_extra(in_data)
        local_feature = self.local(cnn_feature)
        wide_feature = self.wide(cnn_feature)
        return torch.cat([local_feature, wide_feature], dim=1)
class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 7
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 7
            nn.ReLU(),

        )
        self.pre = nn.Sequential(
            nn.Linear(in_features=64*112, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=3),
        )
    def forward(self, feature):
        merge_feature = self.merge(feature)
        out = self.pre(merge_feature.view(merge_feature.size(0), 64*112))
        return out
class CNNd(nn.Module):
    def __init__(self):
        super(CNNd, self).__init__()
        self.feature = CNNdFeature()
        self.predict = Prediction()
    def forward(self, *args):
        return self.predict(self.feature(*args))
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = CNNFeature()
        self.predict = Prediction()
    def forward(self, *args):
        return self.predict(self.feature(*args))
class eTofts(nn.Module):
    def __init__(self,):
        super(eTofts, self).__init__()
        self.deltt = 4.5 / 60
        self.range = torch.arange(0, 112, dtype=torch.float, requires_grad=False)
        self.kt_sin = torch.sin(torch.tensor([alpha,]))
        self.kt_cos = torch.cos(torch.tensor([alpha,]))
    def forward(self, param, T10, cp):
        ktrans, vp, ve = param[:, 0:1]*0.2, param[:, 1:2]*0.1, param[:, 2:3]*0.6
        ktrans = ktrans.clamp(0.00001, 0.2)
        vp = vp.clamp(0.0005, 0.1)
        ve = ve.clamp(0.04, 0.6)
        ce = torch.zeros_like(cp)
        cp_length = cp.size(-1)
        R10 = 1 / T10
        for t in range(cp_length):
            ce[:, t] = torch.sum(cp[:, :t + 1] * torch.exp(ktrans / ve * (self.range[:t + 1] - t) * deltt), dim=1) * deltt
        ce = ce * ktrans
        ct = vp * cp + ce
        R1 = R10 + r1 * ct
        s = (1 - torch.exp(-TR * R1)) * self.kt_sin / (1 - torch.exp(-TR * R1) * self.kt_cos)
        return s*20
    def cuda(self):
        self.range = self.range.cuda()
        self.kt_sin = self.kt_sin.cuda()
        self.kt_cos = self.kt_cos.cuda()
        return self
