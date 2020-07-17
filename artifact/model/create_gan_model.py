from torch import nn

class Generator(nn.Module):
    def __init__(self, noiselen=128, outsize=4, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.outsize = outsize
        self.preprocessor = nn.Sequential(
            nn.Linear(noiselen, 4*outsize*outsize*dim),
            nn.BatchNorm2d(4*outsize*outsize*dim),
            nn.ReLU(True)
        )
        self.block1 = self._make_block(4*dim, 2*dim)
        self.block2 = self._make_block(2*dim, dim)
        self.deconv_out = nn.ConvTranspose2d(dim, 1, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    @classmethod
    def _make_block(cls, indim, outdim, kernel=2, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(indim, outdim, kernel_size=kernel, stride=stride),
            nn.BatchNorm2d(outdim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.preprocessor(x)
        x = x.view((-1,4*self.dim,self.outsize, self.outsize))
        x = self.block1(x)
        x = self.block2(x)
        x = self.deconv_out(x)
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
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
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


