import torch
from torch import nn
from model.GatedConv import GatedConv2dWithActivation as GatedConv2d


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
        # self.lastconv = nn.Conv2d(2,1,kernel_size=1,stride=1)
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                # nn.init.normal_(m.weight,0,0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def _upsample(cls, indim, outdim, kernel=2, stride=2):
        return nn.Sequential(
            # nn.ConvTranspose2d(indim, outdim, kernel_size=kernel, stride=stride),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(indim, outdim, 1),
            nn.BatchNorm2d(outdim),
            # nn.ReLU(True)
        )

    @classmethod
    def _downsample(cls, indim, outdim, kernel=3, stride=2, padding=1):
        return GatedConv2d(indim, outdim, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
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
        self.linear = nn.Linear(4*dim*outsize*outsize, 1)
        self.sigmoid = nn.Sigmoid()
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
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view((-1, 4*self.dim*self.outsize*self.outsize))
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


