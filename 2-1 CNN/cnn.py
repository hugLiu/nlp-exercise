from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3), # 输入通道：1， 输出通道：25， 卷积核大小：3x3
            nn.BatchNorm2d(25), # 然后归一化，防止梯度消失或爆炸
            nn.ReLU(inplace=True) # 激活函数，数据处理，小于0，则等于0，inplace=True节省内存
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) # 池化核大小： 2x2, 步长默认与核大小一致。还有AvgPool2d
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10) # 10分类问题，输出10维
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1) # 维度压缩，flatten the output of conv2 to (batch_size, 50 * 5 * 5)
        x = self.fc(x)
        return x