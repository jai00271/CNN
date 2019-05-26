# Final Program

Here we will try removing the problem with our previous model.

```
# Extract edges and gradients features
model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(28,28,1))) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Trying to extract more features by increasing channels 
model.add(Convolution2D(16, (3, 3), activation='relu')) #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Trying to extract more features by increasing channels 
model.add(Convolution2D(16, (3, 3), activation='relu')) #22

# reducing the size of parameters
model.add(MaxPooling2D(pool_size=(2, 2))) #11

# Since we have done  MP above we should try using 1x1 and fetch co dependend features.
model.add(Convolution2D(16, (1, 1), activation='relu')) #11
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Trying to Increase the channels to fetch parts of object
model.add(Convolution2D(16, (3, 3), activation='relu')) #9
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, (3, 3), activation='relu')) #7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, (3, 3), activation='relu')) #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, (3,3))) #3
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, (3,3))) 

model.add(Flatten())
model.add(Activation('softmax'))
```

Things we modified in the above program are:
1. Added batchNormalization layer before dropout so that we normalize the weights before dropping them.
2. Increased Batch size to 128 from 36 to reduce the epoch time.
