import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
from torch.nn.functional import interpolate

from networks import Flatten, get_pad, GatedConv


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = GatedConv(5, 32, 5, 1, padding=get_pad(256, 5, 1))
        self.layer2 = GatedConv(32, 64, 3, 2, padding=get_pad(256, 4, 2))
        self.layer3 = GatedConv(64, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer4 = GatedConv(64, 128, 3, 2, padding=get_pad(128, 4, 2))
        self.layer5 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer6 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer7 = GatedConv(128, 128, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.layer8 = GatedConv(128, 128, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
        self.layer9 = GatedConv(128, 128, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
        self.layer10 = GatedConv(128, 128, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        self.layer11 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer12 = GatedConv(256, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer13 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        self.layer14 = GatedConv(256, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer15 = GatedConv(128, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        self.layer17 = GatedConv(128, 32, 3, 1, padding=get_pad(256, 3, 1))
        self.layer18 = GatedConv(64, 16, 3, 1, padding=get_pad(256, 3, 1))
        self.layer19 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

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
        out = torch.cat([out, out5], dim=1)
        out = self.layer12(out)
        out = torch.cat([out, out4], dim=1)
        out = self.layer13(out)
        out = self.layer14(out)
        out = torch.cat([out, out3], dim=1)
        out = self.layer15(out)
        out = torch.cat([out, out2], dim=1)
        out = self.layer16(out)
        out = self.layer17(out)
        out = torch.cat([out, out1], dim=1)
        out = self.layer18(out)
        out = self.layer19(out)
        return out


class Generator257(nn.Module):
    def __init__(self):
        super(Generator257, self).__init__()
        self.layer1 = GatedConv(5, 32, 5, 1, padding=get_pad(256, 5, 1))
        self.layer2 = GatedConv(32, 64, 3, 2, padding=get_pad(256, 4, 2))
        self.layer3 = GatedConv(64, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer4 = GatedConv(64, 128, 3, 2, padding=get_pad(128, 4, 2))
        self.layer5 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer6 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer7 = GatedConv(128, 128, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.layer8 = GatedConv(128, 128, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
        self.layer9 = GatedConv(128, 128, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
        self.layer10 = GatedConv(128, 128, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        self.layer11 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer12 = GatedConv(256, 128, 3, 1, padding=get_pad(64, 3, 1))
        # self.layer13 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer14 = GatedConv(256, 64, 3, 1, padding=get_pad(129, 3, 1))
        self.layer15 = GatedConv(128, 64, 3, 1, padding=get_pad(129, 3, 1))
        # self.layer16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer17 = GatedConv(128, 32, 3, 1, padding=get_pad(257, 3, 1))
        self.layer18 = GatedConv(64, 16, 3, 1, padding=get_pad(257, 3, 1))
        self.layer19 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

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
        out = torch.cat([out, out5], dim=1)
        out = self.layer12(out)
        out = torch.cat([out, out4], dim=1)
        # out = self.layer13(out)
        out = interpolate(out, (129, 129), mode='bilinear', align_corners=False)
        out = self.layer14(out)
        out = torch.cat([out, out3], dim=1)
        out = self.layer15(out)
        out = torch.cat([out, out2], dim=1)
        # out = self.layer16(out)
        out = interpolate(out, (257, 257), mode='bilinear', align_corners=False)
        out = self.layer17(out)
        out = torch.cat([out, out1], dim=1)
        out = self.layer18(out)
        out = self.layer19(out)
        # torch.clamp(a, min=-0.5, max=0.5)
        return torch.clamp(out, min=-1, max=1)


class Discriminator256(nn.Module):
    def __init__(self):
        super(Discriminator256, self).__init__()

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
        self.layer10 = nn.utils.spectral_norm(nn.Linear(1000, 256, bias=False))
        self.layer11 = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))

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


class Discriminator257(nn.Module):
    def __init__(self):
        super(Discriminator257, self).__init__()

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
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=0)),
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


class InceptionExtractor256(nn.Module):
    def __init__(self):
        super(InceptionExtractor256, self).__init__()
        # self.inception_v3 = resnet152(pretrained=True)
        self.inception_v3 = inception_v3(pretrained=True, transform_input=True, aux_logits=False)

    def forward(self, x):
        x = interpolate(x, (299, 299), mode='bilinear', align_corners=False)
        x = self.inception_v3((x + 1) / 2)
        x = torch.nn.functional.normalize(x)
        return x


class Discriminator512(nn.Module):
    def __init__(self):
        super(Discriminator512, self).__init__()

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
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 256, bias=False)  # It might not be correct
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
        out = self.layer9(out)
        out_t = self.layer11(out)

        z = self.layer10(z)
        out = (out * z).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out


class InceptionExtractor512(nn.Module):
    def __init__(self):
        super(InceptionExtractor512, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True,  transform_input=True, aux_logits=False)

    def forward(self, x):
        x = self.inception_v3((x + 1) / 2)
        return x
