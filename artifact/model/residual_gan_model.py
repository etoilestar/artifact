import torch
from torch import nn
import model.GatedConv as GatedConv2d
from SN import SwitchNorm2d

class bottleneck(nn.Module):
    def __init__(self, inchannel, channel, reduce=True):
        super(bottleneck, self).__init__()
        self.relu  = nn.ReLU(True)
        self.convin = nn.Conv2d(inchannel, channel, 1, 1)
        self.bnin = nn.BatchNorm2d(channel)
        self.conv = nn.Conv2d(channel, channel, 3, 2, padding=1)
        self.bn = nn.BatchNorm2d(channel)
        self.convout = nn.Conv2d(channel, channel*4, 1, 1)
        self.bnout = nn.BatchNorm2d(channel*4)

        if inchannel == channel * 4:
            self.reduce = False
        else:
            self.reduce = True
        stride = 2 if reduce else 1
        self.reduce_conv = nn.Conv2d(inchannel, channel*4, 1, stride=stride)

    def forward(self, x):
        input = x
        x = self.relu(self.bnin(self.convin(x)))
        x = self.relu(self.bn(self.conv(x)))
        x = self.bnout(self.convout(x))
        if self.reduce:
            x_ = self.reduce_conv(input)
        else:
            x_ = input
        return self.relu(x+x_)



class Generator(nn.Module):
    def __init__(self, noiselen=128, outsize=4, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.outsize = outsize
        # self.preprocessor = nn.Sequential(
        #     nn.Linear(noiselen, 4*outsize*outsize*dim),
        #     nn.BatchNorm2d(4*outsize*outsize*dim),
        #     nn.ReLU(True)
        # )
        self.down1 = self._downsample(1, dim)
        self.down2 = self._downsample(dim, 2*dim)
        self.down3 = self._downsample(2*dim, 4*dim)
        self.down4 = self._downsample(4*dim, 8*dim)
        self.up1 = self._upsample(8*dim, 4*dim)
        self.up2 = self._upsample(4*dim+4*dim, 2*dim)
        self.up3 = self._upsample(2*dim+2*dim, dim)
        self.deconv_out = self._upsample(2*dim, 1)
        # nn.ConvTranspose2d(2*dim, 1, kernel_size=2, stride=2)
        self.lastconv = nn.Conv2d(2,1,kernel_size=1,stride=1)
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                # nn.init.normal_(m.weight,0,0.02)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    @classmethod
    def _upsample(cls, indim, outdim, kernel=2, stride=2):
        return nn.Sequential(
            # nn.ConvTranspose2d(indim, outdim, kernel_size=kernel, stride=stride),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(indim, outdim, 1),
            # nn.BatchNorm2d(outdim),
            SwitchNorm2d(outdim),
            # nn.ReLU(True)
        )

    @classmethod
    def _downsample(cls, indim, outdim, kernel=3, stride=2, padding=1):
        return cls._make_residual_block(indim, outdim//4)
        # nn.Sequential(
            # nn.Conv2d(indim, outdim, kernel_size=kernel, stride=stride, padding=padding),
            # nn.BatchNorm2d(outdim),
        #     SwitchNorm2d(outdim),
        #     nn.ReLU(True)
        # )

    @classmethod
    def _make_residual_block(cls, inchannel, channel):
        return bottleneck(inchannel, channel)

    def forward(self, x):
        # x = self.preprocessor(x)
        # x = x.view((-1,4*self.dim,self.outsize, self.outsize))
        # x = self.block1(x)
        # x = self.block2(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4)
        x6 = self.up2(torch.cat((x5, x3), 1))
        x7 = self.up3(torch.cat((x6, x2), 1))
        x8 = self.deconv_out(torch.cat((x7, x1), 1))
        # x = self.lastconv(torch.cat((x8, x), 1))
        x = self.tanh(x8)
        return x


class Discriminator(nn.Module):
    def __init__(self, indim=1, outsize=4, dim=64):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.outsize = outsize
        self.block1 = self._make_block(indim, dim)
        self.block2 = self._make_block(dim, 2*dim)
        self.block3 = self._make_block(2*dim, 4*dim)
        self.block4 = self._make_block(4*dim, 8*dim)
        self.block5 = self._make_block(8*dim, 16*dim)
        self.block6 = self._make_block(16 * dim, 32 * dim)
        self.linear = nn.Linear(4*dim*outsize*outsize, 1)
        # self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='leaky_relu')
                # nn.init.normal_(m.weight,0,0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def _make_block(cls, indim, outdim, kernel=3, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(indim, outdim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outdim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view((-1, 4*self.dim*self.outsize*self.outsize))
        x = self.linear(x)
        # x = self.sigmoid(x)
        return x


