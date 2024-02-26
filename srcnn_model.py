import torch.nn as nn


# Do not modify this function without guarantee.
def default_conv(in_channels, out_channels, kernel_size, padding, bias=False, init_scale=0.1):
    basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
    basic_conv.weight.data *= init_scale
    if basic_conv.bias is not None:
        basic_conv.bias.data.zero_()
    return basic_conv


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = default_conv(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False, init_scale=0.1)
        self.conv2 = default_conv(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=False, init_scale=0.1)
        self.conv3 = default_conv(in_channels=32, out_channels=3, kernel_size=5, padding=2, bias=False, init_scale=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
        


'''
1.This is VDSR algorithm, but the follow network is not deep as original paper.
2.After you finish the SRCNN model, you can train the VDSR model bellow to observe the 
  difference between two models' result and training process.
3.You can download this paper by searching keywords 'VDSR super resolution' in Google.
'''

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        self.conv1 = default_conv(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.body = nn.Sequential(
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
                nn.ReLU(),
                default_conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, init_scale=0.1),
            )
        self.conv3 = default_conv(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.conv3(out)

        return out + x
