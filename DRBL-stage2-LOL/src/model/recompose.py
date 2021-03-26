# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RECOMPOSE(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RECOMPOSE(nn.Module):
    def __init__(self, args):
        super(RECOMPOSE, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 1
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(args.n_colors*4, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.UPNet = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.UPNet2 = nn.Conv2d(G0, 9, kSize, padding=(kSize-1)//2, stride=1)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x):
        lr = x[0]
        s0 = x[1]
        s1 = x[2]-x[1]
        s2 = x[3]-x[2]

        x = torch.cat((lr, s0, s1, s2), 1)
        original_x = x
        f_1 = self.SFENet1(x)
        x  = self.SFENet2(f_1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x_s1 = self.GFF(torch.cat(RDBs_out,1))
        x = f_1 + x_s1

        weight = self.UPNet2(self.UPNet(x))

        weight = weight*0.8-0.4 + 1
       # weight[weight>1.5] = 1.5
       # weight[weight<0.5] = 0.5

        weight1 = weight[:, 0:3, :, :]
        weight2 = weight[:, 3:6, :, :]
        weight3 = weight[:, 6:9, :, :]

        result = s0 * weight1 + s1 * weight2 + s2 * weight3

        return result, weight1, weight2, weight3
