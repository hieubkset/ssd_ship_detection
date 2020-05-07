import os
import cv2
import xml.etree.ElementTree as ET


def get_bdbox(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()

    obj = root.find('object')
    xml_box = obj.find('bndbox')
    xmin = int(xml_box.find('xmin').text)
    ymin = int(xml_box.find('ymin').text)
    xmax = int(xml_box.find('xmax').text)
    ymax = int(xml_box.find('ymax').text)

    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    gtdir = 'data/VOCdevkit/VOCship/gtimg'
    im_dir = 'data/VOCdevkit/VOCship/JPEGImages'
    an_dir = 'data/VOCdevkit/VOCship/Annotations'

    if not os.path.exists(gtdir):
        os.makedirs(gtdir)

    im_list = os.listdir(im_dir)
    an_list = [im[:-4] + '.xml' for im in im_list]
    im_list = [os.path.join(im_dir, im) for im in im_list]
    an_list = [os.path.join(an_dir, an) for an in an_list]

    for im_file, lb_file in zip(im_list[:], an_list[:]):
        im_name = os.path.basename(im_file)
        dst_save = os.path.join(gtdir, im_name)
        img = cv2.imread(im_file)
        xmin, ymin, xmax, ymax = get_bdbox(lb_file)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.imwrite(dst_save, img)
