import argparse
from datetime import datetime
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import *
from torchsummary import summary

import torch

t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="Pic", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-3, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=500, help="batch interval between model checkpoints")
parser.add_argument("--warmup_epochs", type=int, default=0, help="number of epochs with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=1e-2, help="adversarial loss weight")
parser.add_argument("--save_images", default='images', help="where to store images")
parser.add_argument("--save_models", default='saved_models', help="where to save models")
parser.add_argument("--mask_ratio", type=float, default=0.25, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.save_images, exist_ok=True)
os.makedirs(opt.save_models, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

hr_shape = opt.hr_shape

# Initialize generator and discriminator
if hr_shape == 256:
    from models_256 import *
elif hr_shape == 512:
    from models_512 import *
else:
    print('This input shape is not available')
ie = InceptionExtractor().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Summary of the networks
summary(generator, (5, hr_shape, hr_shape))
summary(discriminator, [(3, hr_shape, hr_shape), (1, hr_shape, hr_shape), (1, 1, 1000)])
summary(ie, (3, hr_shape, hr_shape))

# Set feature extractor to inference mode
ie.eval()

# Losses
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset_name, hr_shape=hr_shape, ratio=opt.mask_ratio),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    D_loss = 0
    G_loss = 0
    content = 0
    adv = 0
    pixel = 0
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        img_alpha = Variable(imgs["alpha"].type(Tensor))
        clip = Variable(imgs['clip'].type(Tensor))
        class_cond = ie(imgs_hr).detach()

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a extended image from input
        gen_hr = generator(clip)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        gen_hr_d = gen_hr * img_alpha + imgs_lr

        if epoch < opt.warmup_epochs:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            pixel += loss_pixel.item()
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr, img_alpha, class_cond).detach()
        pred_fake = discriminator(gen_hr_d, img_alpha, class_cond)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = -pred_fake.mean()

        # Total generator loss
        loss_G = opt.lambda_adv * loss_GAN + loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr, img_alpha, class_cond)
        pred_fake = discriminator(gen_hr_d.detach(), img_alpha, class_cond)

        # Total loss
        loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        D_loss += loss_D.item()
        G_loss += loss_G.item()
        adv += loss_GAN.item()
        pixel += loss_pixel.item()

    avg_D_loss = D_loss / len(dataloader)
    avg_G_loss = G_loss / len(dataloader)
    avg_adv_loss = adv / len(dataloader)
    avg_pixel_loss = pixel / len(dataloader)

    print(
        'Epoch:{1}/{2} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} time:{0}'.format(
            datetime.now() - t_start, epoch + 1, opt.n_epochs, avg_D_loss,
            avg_G_loss, avg_adv_loss, avg_pixel_loss))
    if (epoch + 1) % opt.sample_interval == 0:
        # Save example results
        img_grid = denormalize(torch.cat((imgs_lr, gen_hr, gen_hr_d, imgs_hr), -1))
        save_image(img_grid, opt.save_images + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)
    if (epoch + 1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), opt.save_models + "/generator_{}.pth".format(epoch + 1))
        torch.save(discriminator.state_dict(), opt.save_models + "/discriminator_{}.pth".format(epoch + 1))
