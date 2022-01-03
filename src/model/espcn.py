import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return ESPCN(args)

class ESPCN(nn.Module):
    def __init__(self, args):
        upscale_factor = args.scale[0]
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        #x = torch.tanh(self.conv1(x))
        #x = torch.tanh(self.conv2(x))
        #x = torch.tanh(self.conv3(x))
        #x = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        #return x
        #x = (self.conv1(x))
        #x = (self.conv2(x))
        #x = (self.conv3(x))
        #x = (self.pixel_shuffle(self.conv4(x)))
        #return x
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


if __name__ == "__main__":
    model = Net(upscale_factor=3)
    print(model)
