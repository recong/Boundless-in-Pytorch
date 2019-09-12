import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision.utils import save_image
from torch.autograd import Variable

from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default='Sample/lena_color_512.png')
parser.add_argument("--output", default='output', help="where to save output")
parser.add_argument("--model", default="saved_models/generator_1000.pth", help="generator model pass")
parser.add_argument("--mask_ratio", type=float, default=0.25, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()

os.makedirs(opt.output, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

# Define model and load model checkpoint
generator = Generator().to(device)
generator.load_state_dict(torch.load(opt.model))
generator.eval()

# Prepare input
img = Image.open(opt.image_path)
width, height = img.size
# summary(generator, (4, height, width))
edge = int(width * opt.mask_ratio)
img = np.asarray(img).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
mask = np.zeros((height, width))  # make mask
mask[:, :edge] = 1
mask = mask[np.newaxis, :, :]
img_masked = img * (1 - mask)  # apply mask
img = torch.from_numpy(img).to(device)

mask = torch.from_numpy(mask)
mask_one = torch.ones((height, width), dtype=torch.float64)
img_masked = torch.from_numpy(img_masked)
clip = torch.cat([img_masked, mask_one[None, :, :], mask]).float()
mask = mask.float().to(device)
image_tensor = Variable(clip).to(device).unsqueeze(0)

# Calculate image
with torch.no_grad():
    gen = generator(image_tensor)
    gen_f = gen * mask + img * (1 - mask)
    fn = opt.image_path.split("/")[-1]

    # save_image((img + 1) * 0.5, opt.output + f"/raw.png")
    # save_image((gen + 1) * 0.5, opt.output + f"/gen.png")
    save_image((gen_f + 1) * 0.5, opt.output + f"/gen_{fn}")
    print(f'Output generated image gen_{fn}')
