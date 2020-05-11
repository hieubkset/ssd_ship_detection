# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import logging
import cv2
import os
from mxnet.io import DataBatch, DataDesc
import xml.etree.ElementTree as ET


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """

    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=self.ctx)
        if not isinstance(data_shape, tuple):
            data_shape = (data_shape, data_shape)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[0], data_shape[1]))])
        self.mod.set_params(args, auxs)
        self.mean_pixels = mean_pixels
        self.mean_pixels_nd = mx.nd.array(mean_pixels).reshape((3, 1, 1))

    def create_batch(self, frame):
        """
        :param frame: an (w,h,channels) numpy array (image)
        :return: DataBatch of (1,channels,data_shape,data_shape)
        """
        frame_resize = mx.nd.array(cv2.resize(frame, (self.data_shape[0], self.data_shape[1])))
        # frame_resize = mx.img.imresize(frame, self.data_shape[0], self.data_shape[1], cv2.INTER_LINEAR)
        # Change dimensions from (w,h,channels) to (channels, w, h)
        frame_t = mx.nd.transpose(frame_resize, axes=(2, 0, 1))
        frame_norm = frame_t - self.mean_pixels_nd
        # Add dimension for batch, results in (1,channels,w,h)
        batch_frame = [mx.nd.expand_dims(frame_norm, axis=0)]
        batch_shape = [DataDesc('data', batch_frame[0].shape)]
        batch = DataBatch(data=batch_frame, provide_data=batch_shape)
        return batch

    def detect_iter(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            logging.info("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        result = Detector.filter_positive_detections(detections)
        return result

    def detect_batch(self, batch):
        """
        Return detections for batch
        :param batch:
        :return:
        """
        self.mod.forward(batch, is_train=False)
        detections = self.mod.get_outputs()[0]
        positive_detections = Detector.filter_positive_detections(detections)
        return positive_detections

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect_iter(test_iter, show_timer)
    
    def save_detection(self, save_path, img, boxes, dets, thresh=0.6):
        """
        save detection images

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis('off')
        height = img.shape[0]
        width = img.shape[1]

        for det in dets:
            klass, score, x0, y0, x1, y1 = det
            if score < thresh:
                continue
                
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)

            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor='red',
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 2, '{:.3f}'.format(score),
                           bbox=dict(facecolor='red', alpha=0.5),
                           fontsize=12, color='white')
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor='green',
                                 linewidth=3.5)
            plt.gca().add_patch(rect)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def compute_metrics(self, img, boxes, dets, score_thresh, iou_thresh):
        """
                save detection images

                Parameters:
                ----------
                img : numpy.array
                    image, in bgr format
                dets : numpy.array
                    ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
                    each row is one object
                thresh : float
                    score threshold
                """
        height = img.shape[0]
        width = img.shape[1]
        false_positive = 0
        true_positive = 0
        pred_positive = 0
        positive = len(boxes)
        for det in dets:
            klass, score, x0, y0, x1, y1 = det
            if score < score_thresh:
                continue

            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            pred_box = (xmin, ymin, xmax, ymax)

            pred_positive += 1
            iou = max([self.compute_iou(box, pred_box) for box in boxes])
            if iou >= iou_thresh:
                true_positive += 1
            else:
                false_positive += 1

        return false_positive, true_positive, pred_positive, positive

    @staticmethod
    def filter_positive_detections(detections):
        """
        First column (class id) is -1 for negative detections
        :param detections:
        :return:
        """
        class_idx = 0
        assert (isinstance(detections, mx.nd.NDArray) or isinstance(detections, np.ndarray))
        detections_per_image = []
        # for each image
        for i in range(detections.shape[0]):
            result = []
            det = detections[i, :, :]
            for obj in det:
                if obj[class_idx] >= 0:
                    result.append(obj)
            detections_per_image.append(result)
        logging.info("%d positive detections", len(result))
        return detections_per_image

    @staticmethod
    def get_boxes(label_file):
        tree = ET.parse(label_file)
        root = tree.getroot()
        boxes = []
        for obj in root.iter('object'):
            xml_box = obj.find('bndbox')
            xmin = int(xml_box.find('xmin').text)
            ymin = int(xml_box.find('ymin').text)
            xmax = int(xml_box.find('xmax').text)
            ymax = int(xml_box.find('ymax').text)
            box = (xmin, ymin, xmax, ymax)
            boxes.append(box)
        return boxes

    @staticmethod
    def compute_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def detect(self, im_list, root_dir=None, extension=None,
               classes=[], thresh=0.6, iou=0.75, show_timer=False, save=False):
        """
        wrapper for im_detect and visualize_detection
`
        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        dets = self.im_detect(im_list, os.path.join(root_dir, 'JPEGImages'), extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)

        if save and not os.path.exists('results'):
            os.makedirs('results')

        false_positive, true_positive, pred_positive, positive = 0, 0, 0, 0
        for k, det in enumerate(dets):
            label_file = os.path.join(root_dir, 'Annotations', im_list[k][:-4] + '.xml')
            boxes = self.get_boxes(label_file)
            img = cv2.imread(os.path.join(root_dir, 'JPEGImages', im_list[k]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if save:
                save_path = os.path.join('results', os.path.basename(im_list[k])[:-4] + '.png')
                self.save_detection(save_path, img, boxes, det, thresh)
            metrics = self.compute_metrics(img, boxes, det, thresh, iou)
            false_positive += metrics[0]
            true_positive += metrics[1]
            pred_positive += metrics[2]
            positive += metrics[3]

        recall = true_positive / positive
        precision = true_positive / pred_positive

        print('recall = %.2f' % recall)
        print('precision = %.2f' % precision)
