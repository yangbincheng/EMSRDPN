# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

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

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.multi_scale_infer = args.multi_scale_infer
        self.scale = args.scale

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        ## Up-sampling net
        #if r == 2 or r == 3:
        #    self.UPNet = nn.Sequential(*[
        #        nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
        #        nn.PixelShuffle(r),
        #        nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
        #    ])
        #elif r == 4:
        #    self.UPNet = nn.Sequential(*[
        #        nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
        #        nn.PixelShuffle(2),
        #        nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
        #        nn.PixelShuffle(2),
        #        nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
        #    ])
        #else:
        #    raise ValueError("scale must be 2 or 3 or 4.")

        self.upsample = nn.ModuleList([
            common.Upsampler(common.default_conv, s, G, act=False) for s in args.scale
        ])

        self.tail = nn.ModuleList([
            common.default_conv(G, args.n_colors, kSize) for s in args.scale
        ])


    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        results = []
        if self.multi_scale_infer:
            us_input = x
            for s in range(len(self.scale)):
                x = self.upsample[s](us_input)
                x = self.tail[s](x)
                results.append(x)
            return results
                
        else:
            x = self.upsample[self.scale_idx](x)
            x = self.tail[self.scale_idx](x)
            return x

        #return self.UPNet(x)

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('upsample') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


