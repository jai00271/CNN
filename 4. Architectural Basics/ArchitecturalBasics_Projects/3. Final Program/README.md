# Final Program

Here we will try removing the problem with our previous model.

```
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(64, (3, 3), activation='relu')) #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #12
model.add(Convolution2D(10, (1, 1), activation='relu')) #12
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, (3, 3), activation='relu')) #10
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(64, (3, 3), activation='relu')) #8
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #4
model.add(Convolution2D(10, (4,4))) 
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Activation('softmax'))
```

Things we modified in the above program are:
1. Added batchNormalization layer
2. Increased Batch size to 128 from 36.
