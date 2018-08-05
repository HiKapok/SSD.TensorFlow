# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold, name=None):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope(name, 'select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold

            select_mask = tf.to_float(select_mask)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)

    return selected_bboxes, selected_scores

def clip_bboxes(ymin, xmin, ymax, xmax, height, width, name=None):
    with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, tf.to_float(height) - 1.)
        xmax = tf.minimum(xmax, tf.to_float(width) - 1.)

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)

        return ymin, xmin, ymax, xmax

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name=None):
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin + 1.
        height = ymax - ymin + 1.

        filter_mask = tf.logical_and(width > min_size + 1., height > min_size + 1.)

        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(scores_pred, filter_mask), tf.multiply(ymin, filter_mask), \
                tf.multiply(xmin, filter_mask), tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask)

def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name=None):
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)

        paddings = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(scores, paddings, "CONSTANT"), \
                tf.pad(ymin, paddings, "CONSTANT"), tf.pad(xmin, paddings, "CONSTANT"),\
                tf.pad(ymax, paddings, "CONSTANT"), tf.pad(xmax, paddings, "CONSTANT"),\


def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name=None):
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)

def nms_bboxes_with_padding(scores_pred, bboxes_pred, nms_topk, nms_threshold, name=None):
    with tf.name_scope(name, 'nms_bboxes_with_padding', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        scores = tf.gather(scores_pred, idxes)
        bboxes = tf.gather(bboxes_pred, idxes)

        nms_bboxes = tf.shape(idxes)[0]
        scores_paddings = tf.expand_dims(tf.stack([0, tf.maximum(nms_topk - nms_bboxes, 0)], axis=0), axis=0)
        bboxes_paddings = tf.stack([[0, 0], [tf.maximum(nms_topk - nms_bboxes, 0), 0]], axis=1)

        return tf.pad(scores, scores_paddings, "CONSTANT"), tf.pad(bboxes, bboxes_paddings, "CONSTANT")

def bbox_point2center(bboxes, name=None):
    with tf.name_scope(name, 'bbox_point2center', [bboxes]):
        ymin, xmin, ymax, xmax = tf.unstack(bboxes, 4, axis=-1)
        height, width = (ymax - ymin + 1.), (xmax - xmin + 1.)
        return tf.stack([(ymin + ymax) / 2., (xmin + xmax) / 2., height, width], axis=-1)

def bbox_center2point(bboxes, name=None):
    with tf.name_scope(name, 'bbox_center2point', [bboxes]):
        y, x, h, w = tf.unstack(bboxes, 4, axis=-1)
        return tf.stack([y - (h - 1.) / 2., x - (w - 1.) / 2., y + (h - 1.) / 2., x + (w - 1.) / 2.], axis=-1)

def parse_by_class(image_shape, cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.split(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, image_shape[0], image_shape[1], 'clip_bboxes_{}'.format(class_ind))
            selected_scores[class_ind], ymin, xmin, ymax, xmax = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            selected_scores[class_ind], ymin, xmin, ymax, xmax = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes_with_padding(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_scores
