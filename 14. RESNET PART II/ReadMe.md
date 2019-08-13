## Early Stopping

Keras supports the early stopping of training via a callback called *EarlyStopping*.

This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process.

The *EarlyStopping* callback is configured when instantiated via arguments.

The “*monitor*” allows you to specify the performance measure to monitor in order to end training. Recall from the previous section that the calculation of measures on the validation dataset will have the ‘*val_*‘ prefix, such as ‘*val_loss*‘ for the loss on the validation dataset.

```python
es = EarlyStopping(monitor='val_loss')
```

Often, the first sign of no further improvement may not be the best time to stop training. This is because the model may coast into a plateau of no improvement or even get slightly worse before getting much better.

We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. This can be done by setting the “*patience*” argument.

```python
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
```

The exact amount of patience will vary between models and problems. Reviewing plots of your performance measure can be very useful to get an idea of how noisy the optimization process for your model on your data may be.

By default, any change in the performance measure, no matter how fractional, will be considered an improvement. You may want to consider an improvement that is a specific increment, such as 1 unit for mean squared error or 1% for accuracy. This can be specified via the “*min_delta*” argument.

```python
es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1)
```

Finally, it may be desirable to only stop training if performance stays above or below a given threshold or baseline. For example, if you have familiarity with the training of the model (e.g. learning curves) and know that once a validation loss of a given value is achieved that there is no point in continuing training. This can be specified by setting the “*baseline*” argument.

This might be more useful when fine tuning a model, after the initial wild fluctuations in the performance measure seen in the early stages of training a new model are past.

```python
es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4)
```

## Identifying Overfit models

We can see that expected shape of an overfit model where test accuracy increases to a point and then begins to decrease again.

Reviewing the figure, we can also see flat spots in the ups and downs in the validation loss. Any early stopping will have to account for these behaviors. We would also expect that a good time to stop training might be around epoch 800.

![Line Plots of Loss on Train and Test Datasets While Training Showing an Overfit Model](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/10/Line-Plots-of-Loss-on-Train-and-Test-Datasets-While-Training-Showing-an-Overfit-Model.png)

Line Plots of Loss on Train and Test Datasets While Training Showing an Overfit Model

## Cutout / Random Erasing

Cutout or Random Erasing is a kind of image augmentation methods for convolutional neural networks (CNN). They are very similar methods and were proposed almost at the same time.

They try to regularize models using training images that are randomly masked with random values.

[![img](https://github.com/yu4u/cutout-random-erasing/raw/master/example.png)](https://github.com/yu4u/cutout-random-erasing/blob/master/example.png)

[![img](https://github.com/yu4u/cutout-random-erasing/raw/master/example2.png)](https://github.com/yu4u/cutout-random-erasing/blob/master/example2.png)

## Usage

### With ImageDataGenerator in Keras

It is very easy to use if you are using ImageDataGenerator in Keras; get `eraser` function by `get_random_eraser()`, and then pass it to `ImageDataGenerator` as `preprocessing_function`. By doing so, all images are randomly erased *before* standard augmentation done by ImageDataGenerator.

Please check [cifar10_resnet.py](https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py), which is imported from [official Keras examples](https://github.com/fchollet/keras/tree/master/examples).

What we need to do is add only two lines:

```
...
from random_eraser import get_random_eraser  # added
...

    datagen = ImageDataGenerator(
    ...
        preprocessing_function=get_random_eraser(v_l=0, v_h=1))  # added
```

### Parameters

Parameters are fully configurable as:

```python
get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=255, pixel_level=False)
```

- `p` : the probability that random erasing is performed
- `s_l`, `s_h` : minimum / maximum proportion of erased area against input image
- `r_1`, `r_2` : minimum / maximum aspect ratio of erased area
- `v_l`, `v_h` : minimum / maximum value for erased area
- `pixel_level` : pixel-level randomization for erased area

## Cyclical Learning Rates with Keras and Deep Learning

### What are cyclical learning rates?

[![img](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_triangular.png)](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_triangular.png)

**Figure 1:** Cyclical learning rates oscillate back and forth between two bounds when training, slowly increasing the learning rate after every batch update. To implement cyclical learning rates with Keras, you simply need a callback.

we can define learning rate schedules that monotonically decrease our learning rate after each epoch.

By decreasing our learning rate over time we can allow our model to (ideally) descend into lower areas of the loss landscape.

In practice; however, there are a few problems with a monotonically decreasing learning rate:

- First, our model and optimizer are **still sensitive to our initial choice in learning rate.**
- Second, **we don’t know what the initial learning rate should be** — we may need to perform 10s to 100s of experiments just to find our initial learning rate.
- Finally, there is **no guarantee that our model will descend into areas of low loss**when lowering the learning rate.

To address these issues, Leslie Smith of the NRL introduced **Cyclical Learning Rates** in his 2015 paper, [*Cyclical Learning Rates for Training Neural Networks*](https://arxiv.org/abs/1506.01186).

Now, instead of monotonically decreasing our learning rate, we instead:

1. Define the **lower bound on our learning rate** (called *“base_lr”*).
2. Define the **upper bound on the learning rate** (called the *“max_lr”*).
3. Allow the **learning rate to oscillate back and forth between these two bounds** when training, slowly increasing and decreasing the learning rate *after every batch update*.

An example of a Cyclical Learning Rate can be seen in **Figure 1**.

Notice how our learning rate follows a triangular pattern. First, the learning rate is very small. Then, over time, the learning rate continues to grow until it hits the maximum value. The learning rate then descends back down to the base value. This cyclical pattern continues throughout training.

### Why should we use Cyclical Learning Rates?

[![img](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_saddle_points.png)](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_saddle_points.png)

**Figure 2:** Monotonically decreasing learning rates could lead to a model that is stuck in saddle points or a local minima. By oscillating learning rates cyclically, we have more freedom in our initial learning rate, can break out of saddle points and local minima, and reduce learning rate tuning experimentation. ([image source](https://www.offconvex.org/2016/03/22/saddlepoints/))

As mentioned above, Cyclical Learning Rates enables our learning rate to oscillate back and forth between a lower and upper bound.

So, why bother going through all the trouble?

Why not just monotonically decrease our learning rate, just as we’ve always done?

**The first reason is that our network may become stuck in either saddle points or local minima,** and the low learning rate may not be sufficient to break out of the area and descend into areas of the loss landscape with lower loss.

**Secondly, our model and optimizer may be very sensitive to our initial learning rate choice.** If we make a poor initial choice in learning rate, our model may be stuck from the very start.

**Instead, we can use Cyclical Learning Rates to oscillate our learning rate between upper and lower bounds, enabling us to:**

1. Have more freedom in our initial learning rate choices.
2. Break out of saddle points and local minima.

In practice, using CLRs leads to *far fewer* learning rate tuning experiments along with near identical accuracy to exhaustive hyperparameter tuning.

### How do we use Cyclical Learning Rates?

[![img](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_learning_rate_variations.png)](https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_learning_rate_variations.png)

**Figure 3:** Brad Kenstler’s implementation of deep learning [Cyclical Learning Rates for Keras](https://github.com/bckenstler/CLR) includes three modes — “triangular”, “triangular2”, and “exp_range”. Cyclical learning rates seek to handle training issues when your learning rate is too high or too low shown in this figure. ([image source](https://www.jeremyjordan.me/nn-learning-rate/))

We’ll be using Brad Kenstler’s implementation of [Cyclical Learning Rates for Keras](https://github.com/bckenstler/CLR).

In order to use this implementation we need to define a few values first:

- **Batch size:** Number of training examples to use in a single forward and backward pass of the network during training.
- **Batch/Iteration:** Number of weight updates per epoch (i.e., # of total training examples divided by the batch size).
- **Cycle:** Number of iterations it takes for our learning rate to go from the lower bound, ascend to the upper bound, and then descend back to the lower bound again.
- **Step size:** Number of iterations in a half cycle. Leslie Smith, the creator of CLRs, recommends that the step_size should be (2-8) * training_iterations_in_epoch). **In practice, I have found that step sizes or either 4 or 8 work well in most situations.**

References:

[CutOut](https://github.com/yu4u/cutout-random-erasing)

[Cyclic LR](https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/)

Cyclic LR](https://github.com/bckenstler/CLR)