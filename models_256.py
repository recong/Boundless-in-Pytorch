import torch.nn as nn
import torch
from torchvision.models import resnet152


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ELU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ELU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ELU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ELU(),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer13 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = torch.add(out, out5)
        out = self.layer12(out)
        out = torch.add(out, out4)
        out = self.layer13(out)
        out = self.layer14(out)
        out = torch.add(out, out3)
        out = self.layer15(out)
        out = torch.add(out, out2)
        out = self.layer16(out)
        out = self.layer17(out)
        out = torch.add(out, out1)
        out = self.layer18(out)
        out = self.layer19(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        # self.layer9 = nn.Linear(4096, 256, bias=False)
        self.layer10 = nn.Linear(1000, 256, bias=False)
        self.layer11 = nn.Linear(256, 1, bias=False)

    def forward(self, x, y, z):
        out = torch.cat([x, y], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # out = self.layer9(out)
        out_t = self.layer11(out)

        z = self.layer10(z)
        out = (out * z).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out


class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        self.inception_v3 = resnet152(pretrained=True)

    def forward(self, x):
        x = self.inception_v3(x)
        return x
