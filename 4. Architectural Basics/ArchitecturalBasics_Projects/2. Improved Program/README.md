# Improved Program
We ended our last program with below points to improve:

Things we need to improve in our model:
1. Reduce number of parameteres used in training.
2. Increase model accuracy more.
3.  Reduce training time
4. Too much waiting time to know model accurcy
```
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) #26
model.add(Dropout(0.1))

model.add(Convolution2D(64, (3, 3), activation='relu')) #24
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #12
model.add(Convolution2D(10, (1, 1), activation='relu')) #12
model.add(Dropout(0.1))

model.add(Convolution2D(32, (3, 3), activation='relu')) #10
model.add(Dropout(0.1))

model.add(Convolution2D(64, (3, 3), activation='relu')) #8
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #4
model.add(Convolution2D(10, (4,4))) 
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Activation('softmax'))
```

Reduce number of parameteres used in training is achieved via MaxPooling.
Increase model accuracy more is achieved by DropOut.
Above 2 steps helped in reducing training time.  
Too much waiting time issue resolved by using a callback LearningRateScheduler. Now we know our model accuracy after every epochs.

Now we tried DropOut, Scheduler(to know about model accuracy after every epoch) to achieve the results. 

But, The problem with our model is that we are still using more than 20,000 paramteres and on average training time is more that 10+ sec. 
