train_set_x = Transform(train_set, [Crop(32, 32), FlipLR(), CutOut(8,8)])
train_set_x = train_set_x.map(data_aug)
class specific data augmentation
availability of many augmentation features