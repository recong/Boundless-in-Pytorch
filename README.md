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

## Prepare a dataset
1. Download a dataset ```wget http://data.csail.mit.edu/places/places365/train_256_places365standard.tar```
2. Unpack a tar file ```tar -xvf train_256_places365standard.tar```
3. Run the script using command ```python make_datasets.py```

## Train
1. Run the script using command ```python train.py```

## Test
1. Run the script using command ```python test.py```


In this code, the input size is 512 x 512(in the original paper, 257 x 257).
Due to this change, I intend to align the outputs' sizes of the layers and add an additional layer(layer 9) to the discriminator.

Please let me know if you have any problems.

## 2019/9/13 Update!
Having applied the input size 256 x 256 indicated in the paper, assuming that 257 x 257 is a typo, I noticed some problems as follows:
1. Inception_v3 in pytorch doesn’t support input size 256 x 256; thus, I implemented resnet152 instead. Details are [here](https://discuss.pytorch.org/t/error-in-training-inception-v3/23933/2)
2. In the original paper, the kernel size is 5 x 5 in layer 7. However, this is incorrect since the input size is 4 x 4 so I specified the kernel size 4 x 4 in layer 7.

## 2019/9/26 Update!
Following the author's advice, having applied the input size 257 x 257.
If you want to test the 257 x 257 input, prepare your dataset whose size is 257 x 257 and select it using argparse command ```--dataset_name```