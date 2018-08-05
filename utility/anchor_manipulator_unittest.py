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

import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
import sys; sys.path.insert(0, ".")
from utility import draw_toolbox
from utility import anchor_manipulator
from preprocessing import ssd_preprocessing

slim = tf.contrib.slim

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)

    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
    imsave(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter

def slim_get_split(file_pattern='{}_????'):
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=100,
                items_to_descriptions=None,
                num_classes=21,
                labels_to_names=None)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=2,
                    common_queue_capacity=32,
                    common_queue_min=8,
                    shuffle=True,
                    num_epochs=1)

    [org_image, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'shape',
                                                                         'object/label',
                                                                         'object/bbox',
                                                                         'object/difficult'])
    image, glabels, gbboxes = ssd_preprocessing.preprocess_image(org_image, glabels_raw, gbboxes_raw, [300, 300], is_training=True, data_format='channels_last', output_rgb=True)

    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold = 0.5,
                                                        ignore_threshold = 0.5,
                                                        prior_scaling=[0.1, 0.1, 0.2, 0.2])

    all_anchor_scales = [(30.,), (60.,), (112.5,), (165.,), (217.5,), (270.,)]
    all_extra_scales = [(42.43,), (82.17,), (136.23,), (189.45,), (242.34,), (295.08,)]
    all_anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)]
    all_layer_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    all_layer_strides = [8, 16, 32, 64, 100, 300]
    total_layers = len(all_layer_shapes)
    anchors_height = list()
    anchors_width = list()
    anchors_depth = list()
    for ind in range(total_layers):
        _anchors_height, _anchors_width, _anchor_depth = anchor_encoder_decoder.get_anchors_width_height(all_anchor_scales[ind], all_extra_scales[ind], all_anchor_ratios[ind])
        anchors_height.append(_anchors_height)
        anchors_width.append(_anchors_width)
        anchors_depth.append(_anchor_depth)
    anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask = anchor_encoder_decoder.get_all_anchors([300] * 2, anchors_height, anchors_width, anchors_depth,
                                                                    [0.5] * total_layers, all_layer_shapes, all_layer_strides,
                                                                    [300.] * total_layers, [False] * total_layers)

    gt_targets, gt_labels, gt_scores = anchor_encoder_decoder.encode_anchors(glabels, gbboxes, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask, True)

    num_anchors_per_layer = list()
    for ind, layer_shape in enumerate(all_layer_shapes):
        _, _num_anchors_per_layer = anchor_encoder_decoder.get_anchors_count(anchors_depth[ind], layer_shape)
        num_anchors_per_layer.append(_num_anchors_per_layer)

    # split by layers
    all_anchors = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)

    gt_targets, gt_labels, gt_scores, anchors = tf.split(gt_targets, num_anchors_per_layer, axis=0),\
                                                tf.split(gt_labels, num_anchors_per_layer, axis=0),\
                                                tf.split(gt_scores, num_anchors_per_layer, axis=0),\
                                                tf.split(all_anchors, num_anchors_per_layer, axis=0)

    save_image_op = tf.py_func(save_image_with_bbox,
                            [ssd_preprocessing.unwhiten_image(image),
                            tf.clip_by_value(tf.concat(gt_labels, axis=0), 0, tf.int64.max),
                            tf.concat(gt_scores, axis=0),
                            tf.concat(gt_targets, axis=0)],
                            tf.int64, stateful=True)
    return save_image_op

if __name__ == '__main__':
    save_image_op = slim_get_split('./dataset/tfrecords/train*')
    # Create the graph, etc.
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            print(sess.run(save_image_op))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
