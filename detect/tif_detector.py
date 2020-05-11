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
import rasterio


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

    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels,
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


    def save(self, save_path, img, dets, thresh=0.6):
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

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

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

    def detect(self, image_path, thresh=0.6, iou=0.75):
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
        ds = rasterio.open(image_path)
        band1 = ds.read(1)
        image = np.stack([band1, band1, band1])
        image = image[np.newaxis, :]
        detections = self.mod.predict(image).asnumpy()
        dets = Detector.filter_positive_detections(detections)

        if not isinstance(image_path, list):
            image_path = [image_path]
        assert len(dets) == len(image_path)

        if not os.path.exists('results'):
            os.makedirs('results')

        # for k, det in enumerate(dets):
        #     img = cv2.imread(os.path.join(root_dir, 'JPEGImages', image_path[k]))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     save_path = os.path.join('results', os.path.basename(image_path[k])[:-4] + '.png')
        #     self.save_detection(save_path, img, det, thresh)
