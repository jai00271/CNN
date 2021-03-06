<h3>Architectural Basics</h3>
<ol>

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">How to decide the number of layers in neural network? </span> </li>

This entirely depends upon the size of the input image. Our thought process behind adding layers in our network should solve 2 main problems:<ol>

​		<li>Are we moving towards global minima without putting any overhead on our model. 

​		<li>Is our model effective and optimized in terms of time and memory consumption.</ol>

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">Explain MaxPooling?</span></li>

Let's break the question into 2 parts: <ol>

<li><span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is MaxPooling?</span></li>

MaxPooling concept is used when we have to reduce the number of layers without losing important features in the image.

<li><span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">Why should we use them?</span></li>

When we want to reduce the parameters in our network as with each layer addition to the network we are increasing the parameters and our network is becoming computationally expensive. If we use MP on 400x400 image and use 2x2 matrix to reduce the image, its size will reduce to 200x200 thus reducing many layers as compared to convolution.</ol>

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is 1x1 Convolutions?</span></li>

So we generally increase channel in order 32, 64, 128, 512, 1024, 2048 and reset here to 32 and starts again. But the problem with this approach is that these 32 complex and rich channels which we formed after merging 512 channel needs to remove some feature which is not useful for our network. If we use 3x3 to perform this channel reduction from 512 channel to 32 channel it will re-evaluate and gives new channel, but with 1x1 it will combine the 512 channels and give us 32 channels which won't let the noise such as the background to carry forward. An example would be like if your input image is a face, 3x3 will fetch 2 eyes separately whereas 1x1 will fetch both the eyes saying they always appear together. So when we want to reduce the number of channels we will use 1x1 instead of 3x3. Also a point to remember that 1x1 is computationally very cheap as it is only seeing 1 o/p kernel x n channel at a time instead of 9 o/p kernel x n channel blocks. Check out below image to understand it better.
     ![alt text](https://cdn-images-1.medium.com/max/800/1*HO0_VnNxAYE4k4dblpYzQA.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is 3x3 Convolutions?</span></li>

An image is processed in multiple steps to extract Edges, texture, gradient, parts of object and object itself. Now to move from 1 step to another we need to process the image in such a way that they extract features from the previous layer just like how when we grow in age we use our experience before taking any decision. (read my first [blog](https://dev.to/jai00271/background-basics-21bl) for better understanding)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is Receptive Field?</span></li>

So when we convolve on an image we reduce the image by 2 dimensions. For example, a 5x5 image after one convolution of 3x3 will become 3x3 in size. Now if we convolve on this 3x3 image again it will result in 1x1. Now using this 1x1 image actually I can see my 5x5 image. Let us check out below image and imagine them like a window. If you are standing after the last layer which is layer 3 and we open that 1x1 window you can see 3x3 which is 9 pixels and if I now open that 3x3 window you can see all 5x5 which is 25 pixel. So here the local receptive field for the layer 2 is 3(size of the matrix visible from 1 pixel) and for the layer 3, it will be 5(size of the matrix visible)

![alt text](https://www.researchgate.net/publication/316950618/figure/fig4/AS:495826810007552@1495225731123/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is SoftMax?</span></li>

Softmax assigns decimal probabilities to each class in a multi-class problem. All the probabilities must add up to 1. Softmax is implemented just before the output layer. Softmax layer must have the same number of nodes as the output layer. 

![alt text](https://cdn-images-1.medium.com/max/2000/1*670CdxchunD-yAuUWdI7Bw.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is Learning Rate?</span></li>

LR determines how fast we want to move towards global/ local minima. Less value leads to small steps towards minima causing excessive delay in the process while high value means big steps and you might overshoot and never reach minima.

![alt text](https://cdn-images-1.medium.com/max/1600/0*D7zG46WrdKx54pbU.gif)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What are Kernels and how do we decide the number of kernels?</span></li>

We have multiple options when it comes to kernels like 1x1, 3x3, 11x11. But we prefer 3x3 kernel as it has become standard across the industry(read my first [blog](https://dev.to/jai00271/background-basics-21bl) for better understanding).

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is Batch Normalization?</span></li>

Batch Normalization fixes covariate shift(distribution of feature is different in different parts of training/ test data) in Neural network by normalizing the output of the layers. 

![alt text](https://cdn-images-1.medium.com/max/2000/1*rXY5zJrDdHv6EdKhJvKqcA.png)

The neural network learns by correcting its weight and biases during backpropagation. Now a slight change in starting layer will cause a ripple effect in the next layers which causes a delay in training as the entire network needs to be trained again.
Normalization is a concept of bringing everyone on the same scale to perform the better calculation. For example, if someone asks you how people joined today's meeting, you might reply saying somewhere around 10 people instead of saying 5 men, 5 women joined. 
Batch normalization adds a normalization "layer" between each layers. Normalization has to be done separately for each input neuron over 'mini-batches' and not altogether with all dimensions.

![alt text](https://cdn-images-1.medium.com/max/2000/1*WRio7MD4JDeLww-CyrxEbg.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is Image Normalization?</span></li>

When we are working with images we always won't get the data as we want to train our model. Many time we have to improvise and make changes to the input images and this is called normalization. So according to wiki, Normalization is a process that changes the range of pixel intensity values. Applications include photographs with poor contrast due to glare, for example. Normalization sometimes called contrast or histogram stretching. In more general fields of data processing, such as digital signal processing, it is referred to as dynamic range expansion. The purpose of dynamic range expansion in the various applications is usually to bring the image, or other types of signal, into a range that is more familiar or normal to the senses, hence the term normalization.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is the ideal position of MaxPooling?</span></li>

Maxpooling is preferred minimum after 2 convolution layers. MP helps in reducing the number of parameters in the network.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is the Concept of Transition Layers and what is the position of Transition Layer?</span></li>

Transition layer which is the combination of [convolution + pooling] which is just a way of downsampling the representations calculated by dense blocks to the end as we move from 512x512 to 256x256 to 128x128 and so on. So in simple words decision on reducing/ increasing mathematical complexity of model happens in transition layers.

![alt text](https://i.stack.imgur.com/cSwqp.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is the number of Epochs and when to increase them?</span></li>

How many times the neural network will see the entire dataset is called epochs. For example, if the neural network runs 10 times on 60000 datasets, you will say my epochs are 10. We specify epochs when we train our model like(here epoch is 10):

```
model.fit(X_train, Y_train, nb_epoch=10)
```

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is DropOut?</span></li>

We want our neural network to be effective and the best way to know if your neural network has learned the right features and not memorized the data, then we remove some neurons and run the network again. If results are good that means our network is well trained and if your network performance degrades that means some neuron learned the wrong feature/ memorized data and dictating terms in taking decision by the model. So to summarize, dropout is a regularization technique for reducing overfitting in neural networks.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">When do we introduce DropOut, or when do we know we have some overfitting?</span></li>

When our model is doing well on training data but performing poor on test/ real data we know our model overfits. Some neurons in your network might have learned some feature which might be the deciding factor in image recognition. Due to this your entire network depends upon 1 particular neuron which is wrong. With dropout, we drop some random neurons from the network and retrain the network so that all the neuron are equally and well trained.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What should be the distance of MaxPooling from Prediction layer?</span></li>

MaxPooling should be minimum 2 layers before performing Prediction.

<li> What should be the distance of Batch Normalization from Prediction layer?</li>

Batch normalization should be minimum 1 layer before last convolution layer.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)</span></li>

When we are processing high-quality images where extracting features is computationally very expensive and we can't process image with normal 3x3 convolutions or incase we have performed MaxPooling.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">How do we know our network is not going well, comparatively, very early?</span></li>

When our result after a first epoch is far from target number we know we started wrong and need to change things before retraining. Let us consider MNSIT example. If the result of our first epoch is somewhere around 92% we are targeting 99.5% then we are sure we won't touch 99.5% after n epochs as we have put wrong step forward since beginning.

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is batch size, and effects of batch size?</span></li>

The teacher said you will have 6 test of 1 hour during the whole year and each test will have 30 questions. Here 6 is the epoch and 30 is the batch size. Now the test is of 1 hour and if the teacher gives 120 questions to write, will you be able to do it? It will require you to probably write with both hands :) Let's use the same concept in the neural network. Larger batch size requires more memory power but decrease processing time whereas small batch size means less memory required but more processing time. 

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">When to add validation checks?</span></li>

Ever wondered why we have so many tests before that final exam? It helps you figure out whether you are on the right track or not. When we get fewer marks in a test we know we have a problem here and we need to improve in this area/ layer(neural net). Validation checks are similar to that and we need to know the accuracy of our model after every epoch run instead of knowing it in the end after complete training and waiting for full training to get over to know the accuracy.

<li><span style="background-color: #6a757c; border-left: 2px solid #ffeb8e"> LR schedule and concept behind it?</span></li>

Your learning rate determines how soon you want to reach your minima. If chosen less value the process might take more time to reach minima whereas if the value is chosen more it might overshoot and never reach minima.

![alt text](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)

<li> <span style="background-color: #6a757c; border-left: 2px solid #ffeb8e">What is is the difference between Adam vs SGD?</span></li>

Gradient descent aka batch Gradient Descent is the most common method used to optimize deep learning networks. As per the white paper:

```python
Gradient descent is a way to minimize an objective function J(θ) parameterized by a model’s
parameters θ ∈ (R^d) by updating the parameters in the opposite direction of the gradient of the objective function ∇θJ(θ) w.r.t. to the parameters. The learning rate η determines the size of the steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley

In code, batch gradient descent looks something like this:
for i in range ( nb_epochs ):
    params_grad = evaluate_gradient ( loss_function , data , params )
    params = params - learning_rate * params_grad
```

SGD is a variance of gradient descent. SGD unlike GD performs a paramtere update for *each* training example.

```python
for i in range ( nb_epochs ):
np.random.shuffle ( data )
  for example in data :
     params_grad = evaluate_gradient ( loss_function , example , params )
     params = params - learning_rate * params_grad
```

SGD fluctuation:

![alt text](https://camo.githubusercontent.com/7cafed0adfef0d6261bbd7cf1b66bdc66a707cbb/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f662f66332f53746f6772612e706e67)

Recently many new optimizers became famous and one such optimizer is Adam. Adaptive Moment Estimation(Adam) is another method that compute adaptive learning rates for each parameter. It reaches minima faster than SGD and is also efficient in memory consumption. [Read here for more info](https://stats.stackexchange.com/questions/220494/how-does-the-adam-method-of-stochastic-gradient-descent-work/220563#220563)



[SGD WhitePaper](https://arxiv.org/abs/1609.04747)

[Adam WhitePaper](<https://arxiv.org/abs/1412.6980v8>)

</ol>