create dataset from slices (keep the images ready in the RAM)
shuffle the dataset while picking from the RAM
map the data Augmentations to the image, set num_parallel_calls
create a batch
prefetch 1 batch to make sure that a batch is ready to be served at all times