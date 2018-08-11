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
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import draw_toolbox
from utility import bbox_util

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.2, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size', 4., 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 20, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 200, 'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './logs',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def main(_):
    with tf.Graph().as_default():
        out_shape = [FLAGS.train_image_size] * 2

        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))

        features, output_shape = ssd_preprocessing.preprocess_for_eval(image_input, out_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)
        output_shape = tf.expand_dims(output_shape, axis=0)

        all_anchor_scales = [(30.,), (60.,), (112.5,), (165.,), (217.5,), (270.,)]
        all_extra_scales = [(42.43,), (82.17,), (136.23,), (189.45,), (242.34,), (295.08,)]
        all_anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)]
        # all_anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)]

        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            backbone = ssd_net.VGG16Backbone(FLAGS.data_format)
            feature_layers = backbone.forward(features, training=False)
            with tf.device('/cpu:0'):
                anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold=None, ignore_threshold=None, prior_scaling=[0.1, 0.1, 0.2, 0.2])

                if FLAGS.data_format == 'channels_first':
                    all_layer_shapes = [tf.shape(feat)[2:] for feat in feature_layers]
                else:
                    all_layer_shapes = [tf.shape(feat)[1:3] for feat in feature_layers]
                all_layer_strides = [8, 16, 32, 64, 100, 300]
                total_layers = len(all_layer_shapes)
                anchors_height = list()
                anchors_width = list()
                anchors_depth = list()
                for ind in range(total_layers):
                    _anchors_height, _anchors_width, _anchor_depth = anchor_encoder_decoder.get_anchors_width_height(all_anchor_scales[ind], all_extra_scales[ind], all_anchor_ratios[ind], name='get_anchors_width_height{}'.format(ind))
                    anchors_height.append(_anchors_height)
                    anchors_width.append(_anchors_width)
                    anchors_depth.append(_anchor_depth)
                anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, _ = anchor_encoder_decoder.get_all_anchors(tf.squeeze(output_shape, axis=0),
                                                                                anchors_height, anchors_width, anchors_depth,
                                                                                [0.5] * total_layers, all_layer_shapes, all_layer_strides,
                                                                                [0.] * total_layers, [False] * total_layers)
            location_pred, cls_pred = ssd_net.multibox_head(feature_layers, FLAGS.num_classes, anchors_depth, data_format=FLAGS.data_format)
            if FLAGS.data_format == 'channels_first':
                cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
                location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

            cls_pred = [tf.reshape(pred, [-1, FLAGS.num_classes]) for pred in cls_pred]
            location_pred = [tf.reshape(pred, [-1, 4]) for pred in location_pred]

            cls_pred = tf.concat(cls_pred, axis=0)
            location_pred = tf.concat(location_pred, axis=0)

        with tf.device('/cpu:0'):
            bboxes_pred = anchor_encoder_decoder.decode_anchors(location_pred, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            selected_bboxes, selected_scores = bbox_util.parse_by_class(tf.squeeze(output_shape, axis=0), cls_pred, bboxes_pred,
                                                            FLAGS.num_classes, FLAGS.select_threshold, FLAGS.min_size,
                                                            FLAGS.keep_topk, FLAGS.nms_topk, FLAGS.nms_threshold)

            labels_list = []
            scores_list = []
            bboxes_list = []
            for k, v in selected_scores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
                scores_list.append(v)
                bboxes_list.append(selected_bboxes[k])
            all_labels = tf.concat(labels_list, axis=0)
            all_scores = tf.concat(scores_list, axis=0)
            all_bboxes = tf.concat(bboxes_list, axis=0)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())

            np_image = imread('./demo/test.jpg')
            labels_, scores_, bboxes_, output_shape_ = sess.run([all_labels, all_scores, all_bboxes, output_shape], feed_dict = {image_input : np_image, shape_input : np_image.shape[:-1]})
            bboxes_[:, 0] = bboxes_[:, 0] * np_image.shape[0] / output_shape_[0, 0]
            bboxes_[:, 1] = bboxes_[:, 1] * np_image.shape[1] / output_shape_[0, 1]
            bboxes_[:, 2] = bboxes_[:, 2] * np_image.shape[0] / output_shape_[0, 0]
            bboxes_[:, 3] = bboxes_[:, 3] * np_image.shape[1] / output_shape_[0, 1]

            img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
            imsave('./demo/test_out.jpg', img_to_draw)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
