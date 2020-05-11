import os

image_list = os.listdir('/data/datasets/VOCdevkit/VOC10m/JPEGImages')

os.makedirs('/data/datasets/VOCdevkit/VOC10m/ImageSets/Main')

with open('/data/datasets/VOCdevkit/VOC10m/ImageSets/Main/all.txt', 'w') as f:
    for img in image_list:
        f.write(img[:-4] + '\n')

