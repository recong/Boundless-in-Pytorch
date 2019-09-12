# Boundless: Generative Adversarial Networks for Image Extension in Pytorch

Unofficial pytorch implementation of [Boundless: Generative Adversarial Networks for Image Extension](https://arxiv.org/abs/1908.07007).
I used this code [esrgan](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan) as reference.

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

## Train
1. Set the datasets of images(> 512 x 512) in the folder './Pic'
2. Run the script using command ```python train.py```

## Test
1. Run the script using command ```python test.py```


In this code, the input size is 512 x 512(in original paper, 257 x 257).
Due to this change, I intend to align the outputs' sizes of the layers and add an additional layer(layer 9) to the discriminator.

Please let me know if you have any problems.
