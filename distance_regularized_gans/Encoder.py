import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, ngpu, ngf, nz, nc):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.z_dim = nz
        # input is Z, going into a convolution
        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)
        self.batch1 = nn.BatchNorm2d(ngf)
        self.relu1 = nn.ReLU(True)
        # state size. ``(ngf) x 32 x 32``
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(ngf * 2)
        self.relu2 = nn.ReLU(True)
        # state size. ``(ngf*2) x 16 x 16``
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(ngf * 4)
        self.relu3 = nn.ReLU(True)
        # state size. ``(ngf*4) x 8 x 8``
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(ngf * 8)
        self.relu4 = nn.ReLU(True)
        # state size. ``(ngf*8) x 4 x 4`
        self.conv5 = nn.Conv2d(ngf * 8, nz, 4, 4, 0, bias=False)
        self.batch5 = nn.BatchNorm2d(nz)
        # state size. ``nz``

    def forward(self, input):
        x = self.batch1(self.conv1(input))

        self.relu1(x)
        x = self.batch2(self.conv2(x))
        self.relu2(x)

        x = self.batch3(self.conv3(x))
        self.relu3(x)

        x = self.batch4(self.conv4(x))
        self.relu4(x)

        x = self.conv5(x)

        x = self.batch5(x)

        return x
