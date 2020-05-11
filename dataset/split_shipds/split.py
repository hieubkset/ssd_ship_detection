import os
import random

rootdir = '../../data/VOCdevkit/VOC10m/ImageSets/Main/'

with open(os.path.join(rootdir, 'all.txt'), 'r') as f:
    indices = f.readlines()

l = list(range(len(indices)))
random.shuffle(l)

train_per = 0.7
val_per = 0.2
test_per = 0.1

max_train_idx = int(train_per * len(indices))
max_val_idx = int((train_per + val_per) * len(indices))

train_idx = l[:max_train_idx]
val_idx = l[max_train_idx:max_val_idx]
test_idx = l[max_val_idx:]

with open(rootdir + 'train.txt', 'w') as f:
    for i in train_idx:
        f.write(indices[l[i]])

with open(rootdir + 'val.txt', 'w') as f:
    for i in val_idx:
        f.write(indices[l[i]])

with open(rootdir + 'test.txt', 'w') as f:
    for i in test_idx:
        f.write(indices[l[i]])



