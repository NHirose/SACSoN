#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F

class Sacson_net(nn.Module):
    def __init__(self, in_chs, step_size, lower_bounds, upper_bounds):
        super(Sacson_net, self).__init__()

        dof = len(lower_bounds)
        if len(upper_bounds) != dof:
            raise ValueError("lengths of {lower,upper}_bounds must match")

        lb = torch.Tensor(lower_bounds).repeat((step_size,))[None, :, None, None]
        ub = torch.Tensor(upper_bounds).repeat((step_size,))[None, :, None, None]
        self.register_buffer("out_scale",  (ub - lb) / 2.0)
        self.register_buffer("out_offset", (ub + lb) / 2.0)

        kwargs = {"bn": True, "sample": CBR.DOWN, "activation": self.leaky_relu, "dropout": False}

        self.model_1 = nn.Sequential(
            nn.Conv2d(in_chs, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            CBR( 64, 128, **kwargs),
            CBR(128, 256, **kwargs),
            CBR(256, 512, **kwargs),
            CBR(512, 512, **kwargs),
            CBR(512, 512, **kwargs),
            CBR(512, 512, **kwargs),
            CBR(512, 512, **kwargs),
        )
        self.model_2 = nn.Sequential(
            nn.Linear(513, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, dof * step_size),
            nn.Tanh(),
        )
        self.model_3 = nn.Sequential(
            nn.Linear(512 + 1 + 3*step_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, step_size),
            nn.Sigmoid(),
        )
        self.step_size = step_size

    def forward(self, img, size, fsize, px, pz, ry):
        x1 = self.model_1(img)
        x2 = torch.cat((x1, size.repeat(1,1,1,1)), axis=1)
        x2f = torch.flatten(x2, start_dim=1)
        y = self.model_2(x2f).unsqueeze(2).unsqueeze(3)

        pose_pred = []
        for j in range(self.step_size):
            iom = 0
            while iom < 3:
                pz = pz + y[:, 2*j, 0, 0].detach() * torch.cos(-ry)*0.1
                px = px - y[:, 2*j, 0, 0].detach() * torch.sin(-ry)*0.1
                ry = ry - y[:, 2*j+1, 0, 0].detach() * 0.1
                iom = iom + 1                   

            pose_cat = torch.cat((px.unsqueeze(1), pz.unsqueeze(1), ry.unsqueeze(1)), axis=1)
            pose_pred.append(pose_cat)

        pose_pred = torch.cat(pose_pred, axis=1)

        x3 = torch.cat((x1, fsize.repeat(1,1,1,1), pose_pred.unsqueeze(2).unsqueeze(3)), axis=1)
        x3f = torch.flatten(x3, start_dim=1)
        p = self.model_3(x3f)

        return self.out_scale * y + self.out_offset, p

    def leaky_relu(self, x):
        return F.leaky_relu(x, negative_slope=0.2)
        
class CBR(nn.Module):
    DOWN = True
    UP = False

    def __init__(self, in_channels, out_channels, bn=True, sample=DOWN,
                 activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout

        super(CBR, self).__init__()
        conv = nn.Conv2d if sample == self.DOWN else nn.ConvTranspose2d
        self._conv = conv(in_channels, out_channels, 4, stride=2, padding=1)
        self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self._conv(x)
        if self.bn:
            h = self._bn(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h        
        
