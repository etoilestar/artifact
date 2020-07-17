import torch
from torch import nn

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
        self.up1 = self._upsample(4*dim, 2*dim)
        self.up2 = self._upsample(4*dim, dim)
        self.deconv_out = nn.ConvTranspose2d(2*dim, 1, kernel_size=2, stride=2)
        self.lastconv = nn.Conv2d(2,1,kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def _upsample(cls, indim, outdim, kernel=2, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(indim, outdim, kernel_size=kernel, stride=stride),
            # nn.BatchNorm2d(outdim),
            # nn.ReLU(True)
        )

    @classmethod
    def _downsample(cls, indim, outdim, kernel=3, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(indim, outdim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outdim),
            nn.ReLU(True)
        )

    def forward(self, x):
        # x = self.preprocessor(x)
        # x = x.view((-1,4*self.dim,self.outsize, self.outsize))
        # x = self.block1(x)
        # x = self.block2(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        x5 = self.up2(torch.cat((x4, x2), 1))
        x6 = self.deconv_out(torch.cat((x5, x1), 1))
        x = self.lastconv(torch.cat((x6, x), 1))
        x = self.sigmoid(x)
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='leaky_relu')
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
        return x


