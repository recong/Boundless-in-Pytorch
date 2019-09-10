# Boundless: Generative Adversarial Networks for Image Extension in Pytorch

Unofficial pytorch implementation of [Boundless: Generative Adversarial Networks for Image Extension](https://arxiv.org/abs/1908.07007).
I used this code [esrgan](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan) as reference.
Evaluation mode is coming soon.

## Requirements

```
pytorch
torchvision
torchsummary
numpy
Pillow
random
glob
```

## Usage
1. Set the datasets of images(> 512 x 512) in the folder './Pic'
2. Run the sript using command 'python train.py'


In this code, input size is 512 x 512(in original paper, 257 x 257).
Due to this change, I have to align the outputs' size of the layers and add the additional layer(layer 10) in the discriminator.

Please let me know if you have any problems.
