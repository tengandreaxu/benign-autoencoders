import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.max_pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.gpa = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(in_features=64, out_features=10)

    def forward(self, input):
        # 14 x 14 x 16
        x = self.max_pool(F.relu(self.fc1(input)))
        x = self.batch_norm1(x)

        # 7 x 7 x 32
        x = self.max_pool(F.relu(self.fc2(x)))
        x = self.batch_norm2(x)

        # 3 x 3 x 64
        x = self.max_pool(F.relu(self.fc3(x)))
        x = self.batch_norm3(x)

        # 64
        x = self.gpa(x).view(-1, 64)
        x = self.output(x)
        return x
