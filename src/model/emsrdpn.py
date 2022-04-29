# Efficient Single Image Super-Resolution Using Dual Path Network with Multiple Scale Learning
# https://arxiv.org/

from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return EMSRDPN(args)

class SRDPU(nn.Module):
    def __init__(self, inChannels, numResFeats, numDenFeats, growRate, kSize=3):
        super(SRDPU, self).__init__()
        self.Cin = inChannels
        self.Gr = numResFeats
        self.Gd = numDenFeats
        self.G  = growRate
        self.kSize = kSize

        self.dpb_conv = nn.Sequential(*[
            nn.ReLU(),
            nn.Conv2d(self.Cin, self.Gr+self.G, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.Gr+self.G, self.Gr+self.G, 3, padding=(self.kSize-1)//2, stride=1),
            ])

    def forward(self, x):
        num_of_dense_features = self.Cin - self.Gr
        residual_base, dense_base = torch.split(x, [self.Gr, num_of_dense_features], dim=1)
        dpb_conv = self.dpb_conv(x)
        residual_path, dense_path = torch.split(dpb_conv, [self.Gr, self.G], dim=1)
        residual_sum = torch.add(residual_base, residual_path)
        dense_concat = torch.cat([dense_path, dense_base], 1)
        return torch.cat([residual_sum, dense_concat], 1)

class SRDPB(nn.Module):
    def __init__(self, numResFeats, numDenFeats, growRate, nConvLayers, kSize=3):
        super(SRDPB, self).__init__()
        self.Gr = numResFeats
        self.Gd = numDenFeats
        self.G  = growRate
        self.C  = nConvLayers

        self.dpb_transition = nn.Sequential(*[
            nn.Conv2d(self.Gr+self.Gd+self.C*(self.G), self.Gr+self.Gd, 1, padding=0, stride=1),
            ])

        convs = []
        for c in range(self.C):
            convs.append(SRDPU(self.Gr + self.Gd +  c*self.G, self.Gr, self.Gd, self.G))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        input = x

        x = self.convs(x)
        dpb_transition = self.dpb_transition(x)
        return dpb_transition

class EMSRDPN(nn.Module):
    def __init__(self, args):
        super(EMSRDPN, self).__init__()
        kSize = args.SRDPNkSize
        self.multi_scale_infer = args.multi_scale_infer
        self.scale = args.scale
        
        # Gr, Gd, D, C, G
        self.Gr, self.Gd, self.D, self.C, self.G = {
           'A': (64, 64, 16, 4, 64),
           'B': (137, 0, 16, 4, 0),
           'C': (0, 125, 16, 4, 125),
        }[args.SRDPNconfig]

        self.SFENet1 = nn.Sequential(*[
            nn.Conv2d(args.n_colors, self.Gr+self.Gd, kSize, padding=(kSize-1)//2, stride=1),
            ])

        self.SFENet2 = nn.Sequential(*[
            nn.Conv2d(self.Gr+self.Gd, self.Gr+self.Gd, kSize, padding=(kSize-1)//2, stride=1),
            ])
            
        self.transition = nn.Sequential(*[
            nn.Conv2d(self.D*(self.Gr+self.Gd), self.Gr+self.Gd, 1, padding=0, stride=1),
            nn.Conv2d(self.Gr+self.Gd, self.Gr+self.Gd, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.SRDPBs = nn.ModuleList()
        for i in range(self.D):
            self.SRDPBs.append(
                SRDPB(numResFeats = self.Gr, numDenFeats = self.Gd, growRate = self.G, nConvLayers = self.C)
            )

        self.upsample = nn.ModuleList([
            common.Upsampler(common.default_conv, s, self.Gr+self.Gd, act=False) for s in args.scale
        ])

        self.tail = nn.ModuleList([
            common.default_conv(self.Gr+self.Gd, args.n_colors, kSize) for s in args.scale
        ])

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        SRDPBs_out = []
        for i in range(self.D):
            x = self.SRDPBs[i](x)
            SRDPBs_out.append(x)

        x = torch.cat(SRDPBs_out, 1)
        x = self.transition(x)
        gr = x
        x += f__1
        gi = x
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

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))

