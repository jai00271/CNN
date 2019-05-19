1. How many layers: 

2. MaxPooling: Lets break the question into 3 parts,
	a. What is MaxPooling?
		MaxPooling concept is used when we have to reduce the number of layers without losing important important features in the image.
	b. Why we should we use them?

	c. When should we use them?
		If we use MP on 400x400 image, its size will reduce to 200x200 thus reducing many layers as compared to convolution.
 
3. 1x1 Convolutions
	So we generally increase channel in order 32, 64, 128, 512, 1024, 2048 and reset here to 32 and starts again. But the problem with this approach is these 32 complex and rich channels which we formed after merging 512 channel needs to remove some feature which are not useful for our network. If we use 3x3 to perform this channel reduction from 512 channel to 32 channel it will re-evaluate and gives new channel, but with 1x1 it will combine the 512 channels and give us 32 channels which won't let the noise such as background to carry forward. Example would be like if you input image is a face, 3x3 will fetch 2 eyes separately whereas 1x1 will fetch both the eyes saying they always appear together. So when we want to reduce number of channels we will use 1x1 instead of 3x3. Also a point to remeber that 1x1 is computationally very cheap as it is only seeing 1 o/p kernel x n channel at a time instead of 9 o/p kernel x n channel blocks. Check out below image to understand it better.
	![1x1 convolution]("https://cdn-images-1.medium.com/max/800/1*HO0_VnNxAYE4k4dblpYzQA.png")

4. 3x3 Convolutions
	An image is processed in multiple steps
Receptive Field,
SoftMax,
Learning Rate,
Kernels and how do we decide the number of kernels?
Batch Normalization,
Image Normalization,
Position of MaxPooling,
Concept of Transition Layers,
Position of Transition Layer,
Number of Epochs and when to increase them,
DropOut
When do we introduce DropOut, or when do we know we have some overfitting
The distance of MaxPooling from Prediction,
The distance of Batch Normalization from Prediction,
When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
How do we know our network is not going well, comparatively, very early
Batch Size, and effects of batch size
When to add validation checks
LR schedule and concept behind it
Adam vs SGD
etc (you can add more if we missed it here)