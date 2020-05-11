import os
import cv2

dir = '/data/datasets/VOCdevkit/VOC10m/JPEGImages'
image_list = os.listdir(dir)
for i in image_list:
    img = cv2.imread(os.path.join(dir, i))
    try:
        m = min(img.shape)
    except Exception:
        print(i)