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

import numpy as np
from scipy.misc import imread, imsave, imshow, imresize

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import scaffolds
from utility import bbox_util
from utility import draw_toolbox

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', './dataset/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for training and evaluation.')
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
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size', 4., 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 400, 'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % FLAGS.model_dir)
        return None

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image).astype(np.uint8)

    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
    imsave(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter

# couldn't find better way to pass params from input_fn to model_fn
# some tensors used by model_fn must be created in input_fn to ensure they are in the same graph
# but when we put these tensors to labels's dict, the replicate_model_fn will split them into each GPU
# the problem is that they shouldn't be splited
global_anchor_info = dict()

def input_pipeline(dataset_pattern='val-*', is_training=True, batch_size=FLAGS.batch_size):
    def input_fn():
        assert batch_size==1, 'We only support single batch when evaluation.'
        target_shape = [FLAGS.train_image_size] * 2
        image_preprocessing_fn = lambda image_, labels_, bboxes_ : ssd_preprocessing.preprocess_image(image_, labels_, bboxes_, target_shape, is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)

        image, filename, shape, output_shape = dataset_common.slim_get_batch(FLAGS.num_classes,
                                                                            batch_size,
                                                                            ('train' if is_training else 'val'),
                                                                            os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                            FLAGS.num_readers,
                                                                            FLAGS.num_preprocessing_threads,
                                                                            image_preprocessing_fn,
                                                                            None,
                                                                            num_epochs=1,
                                                                            is_training=is_training)

        return {'image': image, 'filename': filename, 'shape': shape, 'output_shape': output_shape}, None
    return input_fn

def ssd_model_fn(features, labels, mode, params):
    """model_fn for SSD to be used with our Estimator."""
    filename = features['filename']
    filename = tf.identity(filename, name='filename')
    shape = features['shape']
    output_shape = features['output_shape']
    features = features['image']

    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold=None, ignore_threshold=None, prior_scaling=[0.1, 0.1, 0.2, 0.2])
    all_anchor_scales = [(30.,), (60.,), (112.5,), (165.,), (217.5,), (270.,)]
    all_extra_scales = [(42.43,), (82.17,), (136.23,), (189.45,), (242.34,), (295.08,)]
    all_anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)]
    #all_anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)]

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net.VGG16Backbone(params['data_format'])
        # forward features
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        # generate anchors according to the feature map size
        with tf.device('/cpu:0'):
            if params['data_format'] == 'channels_first':
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
        # generate predictions based on anchors
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers, params['num_classes'], anchors_depth, data_format=params['data_format'])
        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]

        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)

        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])
    # decode predictions
    with tf.device('/cpu:0'):
        bboxes_pred = anchor_encoder_decoder.decode_anchors(location_pred, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
        selected_bboxes, selected_scores = bbox_util.parse_by_class(tf.squeeze(output_shape, axis=0), cls_pred, bboxes_pred,
                                                        params['num_classes'], params['select_threshold'], params['min_size'],
                                                        params['keep_topk'], params['nms_topk'], params['nms_threshold'])

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
    save_image_op = tf.py_func(save_image_with_bbox,
                        [ssd_preprocessing.unwhiten_image(tf.squeeze(features, axis=0), output_rgb=False),
                        all_labels * tf.to_int32(all_scores > 0.3),
                        all_scores,
                        all_bboxes],
                        tf.int64, stateful=True)
    tf.identity(save_image_op, name='save_image_op')
    predictions = {'filename': filename, 'shape': shape, 'output_shape': output_shape }
    for class_ind in range(1, params['num_classes']):
        predictions['scores_{}'.format(class_ind)] = tf.expand_dims(selected_scores[class_ind], axis=0)
        predictions['bboxes_{}'.format(class_ind)] = tf.expand_dims(selected_bboxes[class_ind], axis=0)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              prediction_hooks=None, loss=None, train_op=None)
    else:
        raise ValueError('This script only support "PREDICT" mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=None).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    summary_dir = os.path.join(FLAGS.model_dir, 'predict')
    tf.gfile.MakeDirs(summary_dir)
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'select_threshold': FLAGS.select_threshold,
            'min_size': FLAGS.min_size,
            'nms_threshold': FLAGS.nms_threshold,
            'nms_topk': FLAGS.nms_topk,
            'keep_topk': FLAGS.keep_topk,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
        })
    tensors_to_log = {
        'cur_image': 'filename',
        'cur_ind': 'save_image_op'
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting a predict cycle.')
    pred_results = ssd_detector.predict(input_fn=input_pipeline(dataset_pattern='val-*', is_training=False, batch_size=FLAGS.batch_size),
                                    hooks=[logging_hook], checkpoint_path=get_checkpoint())#, yield_single_examples=False)

    det_results = list(pred_results)
    #print(list(det_results))

    #[{'bboxes_1': array([[0.        , 0.        , 284.59054, 567.9505 ], [31.58835 , 34.792888, 73.12541 , 100.        ]], dtype=float32), 'scores_17': array([0.01333667, 0.01152573], dtype=float32), 'filename': b'000703.jpg', 'shape': array([334, 500,   3])}]
    for class_ind in range(1, FLAGS.num_classes):
        with open(os.path.join(summary_dir, 'results_{}.txt'.format(class_ind)), 'wt') as f:
            for image_ind, pred in enumerate(det_results):
                filename = pred['filename']
                shape = pred['shape']
                output_shape = pred['output_shape']
                scores = pred['scores_{}'.format(class_ind)]
                bboxes = pred['bboxes_{}'.format(class_ind)]
                bboxes[:, 0] = bboxes[:, 0] * shape[0] / output_shape[0]
                bboxes[:, 1] = bboxes[:, 1] * shape[1] / output_shape[1]
                bboxes[:, 2] = bboxes[:, 2] * shape[0] / output_shape[0]
                bboxes[:, 3] = bboxes[:, 3] * shape[1] / output_shape[1]

                valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 1.), (bboxes[:, 3] - bboxes[:, 1] > 1.))

                for det_ind in range(valid_mask.shape[0]):
                    if not valid_mask[det_ind]:
                        continue
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(filename.decode('utf8')[:-4], scores[det_ind],
                                       bboxes[det_ind, 1], bboxes[det_ind, 0],
                                       bboxes[det_ind, 3], bboxes[det_ind, 2]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs('./debug')
  tf.app.run()
