import os
import random

with open('trainval.txt', 'r') as f:
    indices = f.readlines()

l = list(range(len(indices)))
random.shuffle(l)

train_per = 0.7
test_per = 0.3

max_train_idx = int(0.7 * len(indices))

train_idx = l[:max_train_idx]
test_idx = l[max_train_idx:]

with open('../../data/VOCdevkit/VOCship/ImageSets/Main/trainval.txt', 'w') as f:
    for i in train_idx:
        f.write(indices[l[i]])

with open('../../data/VOCdevkit/VOCship/ImageSets/Main/test.txt', 'w') as f:
    for i in test_idx:
        f.write(indices[l[i]])



