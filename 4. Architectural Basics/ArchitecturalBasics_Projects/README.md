# Vanilla Program
Here we will try minimal configuration to our existing program and try to understand how something works and what are its disadvantage.
In this program we started with a curtain on our mind of creativity and used straight forward to acheive our target. 

```
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) #26
model.add(Convolution2D(64, (3, 3), activation='relu')) #24
model.add(Convolution2D(128, (3, 3), activation='relu')) #22
model.add(Convolution2D(256, (3, 3), activation='relu')) #20
model.add(Convolution2D(512, (3, 3), activation='relu')) #18
model.add(MaxPooling2D(pool_size=(2, 2))) #9
model.add(Convolution2D(32, (3,3), activation='relu')) #7
model.add(Convolution2D(64, (3, 3), activation='relu')) #5
model.add(Convolution2D(10, (5,5))) 
model.add(Flatten())
model.add(Activation('softmax'))
```
We convolve on our training set using 3x3 kernel and when  we reached 512 channel we used MaxPooling to reduce the size of image as our parameters were inceasing quite fast. Now if you notice in the pragram we have used around 1,749,994 params which is quite bad to train a image of size 28x28. Something we achived here:
1. Convlution using 3x3 
2. Used MaxPooling 
3. Used Flatten
4. Softmax Activation funtion
5. Reached model accuracy of 0.99

Things we need to improve in our model:
1. Reduce number of parameteres used in training.
2. Increase model accuracy more.
3.  Reduce training time
