import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision.utils import save_image
from torch.autograd import Variable

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default='Sample/lena_color_512.png')
parser.add_argument("--output", default='output', help="where to save output")
parser.add_argument("--checkpoint_model", default="saved_models_3/generator_2000.pth", help="generator model pass")
parser.add_argument("--mask_ratio", type=float, default=0.25, help="the ratio of pixels in the image are respectively masked")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()

os.makedirs("output", exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

# Define model and load model checkpoint
generator = Generator().to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
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
img_masked = torch.from_numpy(img_masked)
clip = torch.cat([img_masked, mask]).float()
mask = mask.float().to(device)
image_tensor = Variable(clip).to(device).unsqueeze(0)

# Calculate image
with torch.no_grad():
    gen_hr = generator(image_tensor)
    gen_hr = gen_hr * mask + img
    img_grid = denormalize(gen_hr)
    fn = opt.image_path.split("/")[-1]
    save_image(img_grid, opt.output + f"/{fn}")
    print(f'Output generated image {fn}')
