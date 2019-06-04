## Advanced Convolutions

*Dilated Convolution a.k.a Atrous Convolution*

Dilated convolutions helps us in increasing the receptive field without adding any parameters to our network. The key application the dilated convolution authors have in mind is dense prediction: vision applications where the predicted object that has similar size and structure to the input image. Dilated convolutions are used in Image segmentations, Audio generations, Machine translations, etc. In many such applications one wants to integrate information from different spatial scales and balance two properties:

1. local, pixel-level accuracy, such as precise detection of edges, and
2. integrating knowledge of the wider, global context

Simply putting,

**When l=1, it is standard convolution.**

**When l>1, it is dilated convolution.**


  ![alt text](https://cdn-images-1.medium.com/max/2000/0*oX5IPr7TlVM2NpEU.gif)

**Standard Convolution (l=1)**

![alt text](https://cdn-images-1.medium.com/max/2000/0*3cTXIemm0k3Sbask.gif)

**Dilated Convolution (l=2)**

We can see that **the receptive field is larger** compared with the standard one.

![alt text](https://cdn-images-1.medium.com/max/2000/1*tnDNIyPePgHvb8JIx8SbqA.png)

​                            (a)                                                            (b)                                              (c)

In above image we see that Systematic dilation supports exponential expansion of the receptive field without loss of
resolution or coverage. (a) F1 is produced from F0 by a 1-dilated convolution; each element in F1 has a receptive field of 3×3. (b) F2 is produced from F1 by a 2-dilated convolution; each element in F2 has a receptive field of 7×7. (c) F3 is produced from F2 by a 4-dilated convolution; each element in F3 has a receptive field of 15×15. The number of parameters associated with each layer is identical. The receptive field grows exponentially while the number of parameters grows linearly. easy to see that the size of the receptive field of each element in Fi+1 is (2i+2 − 1)×(2i+2 − 1).
The receptive field is a square of exponentially increasing size.

------

*DECONVOLUTION or Fractionally Strided OR Transpose Convolution*

The improvement of resolution of images or other data by a mathematical algorithm designed to separate the information from artefacts which result from the method of collecting it. Deconvolution layer is a very unfortunate name and should rather be called a transposed convolutional layer.

Visually, for a transposed convolution with stride one and no padding, we just pad the original input (blue entries) with zeroes (white entries) (Figure 1).

![alt text](https://i.stack.imgur.com/YyCu2.gif)

In case of stride two and padding, the transposed convolution would look like this (Figure 2):

![alt text](https://i.stack.imgur.com/f2RiP.gif)

You can find more (great) visualisations of convolutional arithmetics [here](https://github.com/vdumoulin/conv_arithmetic).

Reference:

[Dilated Convo White paper](https://arxiv.org/pdf/1511.07122.pdf)

[DECONVOLUTION or Fractionally Strided OR Transpose Convolution](https://datascience.stackexchange.com/a/12110/74860)

