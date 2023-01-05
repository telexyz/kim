# Needle vs PyTorch

## Initialization is matter

Torch intialize Conv and Linear layers' weights according to non-linear activation function (relu vs leaky_relu). When implement the model in Needle, we did not aware that then use the only and default relu nonlinearity config. That cause the final loss of needle model always larger than torch final loss. Needle losses around 1.0, torch around 0.7, about 0.3 different.

To find out exactly which compoment of the network cause the different. We run both torch model and needle model on the same mini-batches, hyper-paramaters, and configs. We even created a PseudoDropout layer in needle to copy exactly torch Dropout layer behavior.

We run a mini-experience by training both models on 16k input images, for 5 epochs, and compare losses at the end of each epoch. In the first run, we copy needle initialized weights to torch, then the losses are exactly the same, so we are confident that our implement are correct and needle accuracy is on-par with torch.
```
loss         needle     torch          diff
epoch 0: 0.96261054 0.9625967 1.3828278e-05
epoch 1: 1.050356   1.0503557 3.5762787e-07
epoch 2: 1.1356571  1.1356555 1.5497208e-06
epoch 3: 1.0025123  1.0025071 5.2452087e-06
epoch 4: 0.7895985  0.7896045 5.9604645e-06
```

For the second run, we force needle to mimic torch dropout but don't copy needle initialized weights to torch.
```
loss         needle      torch       diff
epoch 0: 1.1341821  0.797671   0.33651108
epoch 1: 1.0557958  0.7406452  0.31515062
epoch 2: 1.1040108  0.74287474 0.36113608
epoch 3: 1.0368311  0.7846966  0.25213456
epoch 4: 0.97958153 0.762485   0.21709651
```

For the third run, we use same weight initialization, but don't force needle to mimic torch dropout. Here is the result:
```
loss        needle     torch         diff
epoch 0: 1.1006815 1.1059185 0.0052369833
epoch 1: 1.0643365 1.0856049 0.021268368
epoch 2: 1.1491817 1.1492293 0.000047564
epoch 3: 1.2083136 1.0732015 0.13511205
epoch 4: 1.1208873 1.0828264 0.038060904
```

We can conclude from 2nd run and 3rd run that the main source of different is on initialization. We scanned the torch Linear and Conv2d source code and find out that they use a different kaiming_uniform initialization paramaters for leaky_relu. We followed torch source code to initialize needle Conv and Linear layers. The final results is better training / validation losses, and similar to torch:

Final comparision: training 600k images, validating 100k images, testing 300k images, 160 images / mini-batch, 5 epoches.

```
 NEEDLE
- - - -
[ train ] Epoch: 0 Batch: 3749 Loss: 0.7249: 100%|██████████████████████████████████| 3750/3750 [54:46<00:00,  1.14it/s]
[ valid ] Epoch: 0 Batch:  624 Loss: 0.6920: 100%|██████████████████████████████████|   625/625 [02:28<00:00,  4.21it/s]
[ train ] Epoch: 1 Batch: 3749 Loss: 0.7050: 100%|██████████████████████████████████| 3750/3750 [44:33<00:00,  1.40it/s]
[ valid ] Epoch: 1 Batch:  624 Loss: 0.6936: 100%|██████████████████████████████████|   625/625 [02:34<00:00,  4.03it/s]
[ train ] Epoch: 2 Batch: 3749 Loss: 0.7001: 100%|██████████████████████████████████| 3750/3750 [53:19<00:00,  1.17it/s]
[ valid ] Epoch: 2 Batch:  624 Loss: 0.6974: 100%|██████████████████████████████████|   625/625 [02:29<00:00,  4.18it/s]
[ train ] Epoch: 3 Batch: 3749 Loss: 0.6971: 100%|██████████████████████████████████| 3750/3750 [45:02<00:00,  1.39it/s]
[ valid ] Epoch: 3 Batch:  624 Loss: 0.6909: 100%|██████████████████████████████████|   625/625 [02:29<00:00,  4.18it/s]
[ train ] Epoch: 4 Batch: 3749 Loss: 0.6956: 100%|██████████████████████████████████| 3750/3750 [44:05<00:00,  1.42it/s]
[ valid ] Epoch: 4 Batch:  624 Loss: 0.6964: 100%|██████████████████████████████████|   625/625 [02:32<00:00,  4.09it/s]
[ test ] Acc: 50.4% Loss: 0.6939: 100%|█████████████████████████████████████████████| 1875/1875 [07:23<00:00,  4.23it/s]

TORCH
- - -
[ train ] Epoch: 1 Batch: 3749 Loss: 0.7130: 100%|██████████████████████████████████| 3750/3750 [08:28<00:00,  7.37it/s]
[ valid ] Epoch: 1 Batch:  624 Loss: 0.7134: 100%|██████████████████████████████████|   625/625 [01:39<00:00,  6.28it/s]
[ train ] Epoch: 2 Batch: 3749 Loss: 0.7031: 100%|██████████████████████████████████| 3750/3750 [08:18<00:00,  7.52it/s]
[ valid ] Epoch: 2 Batch:  624 Loss: 0.7015: 100%|██████████████████████████████████|   625/625 [01:42<00:00,  6.10it/s]
[ train ] Epoch: 3 Batch: 3749 Loss: 0.6979: 100%|██████████████████████████████████| 3750/3750 [08:23<00:00,  7.45it/s]
[ valid ] Epoch: 3 Batch:  624 Loss: 0.6979: 100%|██████████████████████████████████|   625/625 [01:40<00:00,  6.20it/s]
[ train ] Epoch: 4 Batch: 3749 Loss: 0.6952: 100%|██████████████████████████████████| 3750/3750 [08:27<00:00,  7.38it/s]
[ valid ] Epoch: 4 Batch:  624 Loss: 0.6988: 100%|██████████████████████████████████|   625/625 [01:41<00:00,  6.16it/s]
[ test ] Acc: 50.9% Loss: 0.6974: 100%|█████████████████████████████████████████████| 1875/1875 [05:04<00:00,  6.16it/s]
```

## Performance and efficiency
From training logs above, we can easily see that torch training is ~6x faster than needle, and torch inference is ~1.5x faster than needle.

In term of efficiency, we noticed that needle utilize roughly double GPU power and a bit more GPU memory.

One thing to notice is that torch utilize much more CPU and RAM on host machine than needle. Torch use 184% of CPU and 3.1G of RAM, meanwhile needle use 86% of CPU and 0.9G of RAM.

We guest that, torch superior is due to it's better tensor operation implementation (matrix multiplication, convolution), better data-pilining, and better computational graph optimization for backward computation. Better graph optimization should be the main reason that lead to torch training speed is ~6x higher than needle, meanwhile inference speed is only ~1.5x.

![](files/needle_vs_torch-00.png)
_needle training GPU utilization_

![](files/needle_vs_torch-02.png)
_torch training GPU utilization_

- - -

The original paper use Xavier initialization, I missed that part when reading the paper.
![](files/needle_vs_torch-01.png)

