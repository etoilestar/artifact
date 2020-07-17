from torch import nn

class MYmodel(nn.Module):
    def __init__(self):#定义网络层结构，类似卷积，线性层，激活层等
        super(MYmodel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):#前向传播
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.conv3(x)
