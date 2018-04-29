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

#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 16,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,
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
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.6, 'Matching threshold in the loss function.')#0.6
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.4, 'Matching threshold for the negtive examples in the loss function.')#0.4
# optimizer related configuration
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0002, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')#0.001
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '70000, 90000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '1, 0.8, 0.1',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ssd300/xdet_head, ssd300/xdet_multi_path, ssd300/xdet_additional_conv',#None
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')

FLAGS = tf.app.flags.FLAGS

def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                        FLAGS.checkpoint_model_scope, FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars)

def input_pipeline():
    image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
        'xdet_resnet', is_training=True)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'))

    anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                    layers_shapes = [(38, 38)],
                                                    anchor_scales = [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                                                    extra_anchor_scales = [[0.1]],
                                                    anchor_ratios = [[1., 2., 3., .5, 0.3333]],
                                                    layer_steps = [8])

    def input_fn():
        all_anchors, num_anchors_list = anchor_creator.get_all_anchors()

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(all_anchors,
                                        num_classes = FLAGS.num_classes,
                                        allowed_borders = [0.05],
                                        positive_threshold = FLAGS.match_threshold,
                                        ignore_threshold = FLAGS.neg_threshold,
                                        prior_scaling=[0.1, 0.1, 0.2, 0.2])
        list_from_batch, _ = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                FLAGS.dataset_split_name,
                                                FLAGS.data_dir,
                                                image_preprocessing_fn,
                                                file_pattern = None,
                                                reader = None,
                                                batch_size = FLAGS.batch_size,
                                                num_readers = FLAGS.num_readers,
                                                num_preprocessing_threads = FLAGS.num_preprocessing_threads,
                                                num_epochs = FLAGS.train_epochs,
                                                anchor_encoder = anchor_encoder_decoder.encode_all_anchors)

        return list_from_batch[-1], {'targets': list_from_batch[:-1],
                                    'decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors([pred])[0],
                                    'num_anchors_list': num_anchors_list}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights = 1., bbox_outside_weights = 1., sigma = 1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul

def xdet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    num_anchors_list = labels['num_anchors_list']
    num_feature_layers = len(num_anchors_list)

    shape = labels['targets'][-1]
    glabels = labels['targets'][:num_feature_layers][0]
    gtargets = labels['targets'][num_feature_layers : 2 * num_feature_layers][0]
    gscores = labels['targets'][2 * num_feature_layers : 3 * num_feature_layers][0]

    with tf.variable_scope(params['model_scope'], default_name = None, values = [features], reuse=tf.AUTO_REUSE):
        backbone = xdet_body_v2.xdet_resnet_v2(params['resnet_size'], params['data_format'])
        body_cls_output, body_regress_output = backbone(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

        cls_pred, location_pred = xdet_body_v2.xdet_head(body_cls_output, body_regress_output, params['num_classes'], num_anchors_list[0], (mode == tf.estimator.ModeKeys.TRAIN), data_format=params['data_format'])

    if params['data_format'] == 'channels_first':
        cls_pred = tf.transpose(cls_pred, [0, 2, 3, 1])
        location_pred = tf.transpose(location_pred, [0, 2, 3, 1])

    bboxes_pred = labels['decode_fn'](location_pred)#(tf.reshape(location_pred, tf.shape(location_pred).as_list()[0:-1] + [-1, 4]))

    cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
    location_pred = tf.reshape(location_pred, [-1, 4])
    glabels = tf.reshape(glabels, [-1])
    gscores = tf.reshape(gscores, [-1])
    gtargets = tf.reshape(gtargets, [-1, 4])

    # raw mask for positive > 0.5, and for negetive < 0.3
    # each positive examples has one label
    positive_mask = glabels > 0#tf.logical_and(glabels > 0, gscores > params['match_threshold'])
    fpositive_mask = tf.cast(positive_mask, tf.float32)
    n_positives = tf.reduce_sum(fpositive_mask)
    # negtive examples are those max_overlap is still lower than neg_threshold, note that some positive may also has lower jaccard
    # note those gscores is 0 is either be ignored during anchors encode or anchors have 0 overlap with all ground truth
    #negtive_mask = tf.logical_and(tf.logical_and(tf.logical_not(tf.logical_or(positive_mask, glabels < 0)), gscores < params['neg_threshold']), gscores > 0.)
    #gscores = tf.Print(gscores, [tf.reduce_sum(tf.cast(gscores > 0., tf.float32))])
    #glabels = tf.Print(glabels, [glabels, tf.reduce_sum(tf.cast(tf.equal(glabels, 0), tf.float32))], message='glabels: ', summarize=1000)

    negtive_mask = tf.logical_and(tf.equal(glabels, 0), gscores > 0.)
    #negtive_mask = tf.Print(negtive_mask, [tf.reduce_sum(tf.cast(negtive_mask, tf.float32))])
    #negtive_mask = tf.logical_and(tf.logical_and(tf.logical_not(positive_mask), gscores < params['neg_threshold']), gscores > 0.)
    #negtive_mask = tf.logical_and(gscores < params['neg_threshold'], tf.logical_not(positive_mask))
    fnegtive_mask = tf.cast(negtive_mask, tf.float32)
    n_negtives = tf.reduce_sum(fnegtive_mask)

    n_neg_to_select = tf.cast(params['negative_ratio'] * n_positives, tf.int32)
    n_neg_to_select = tf.minimum(n_neg_to_select, tf.cast(n_negtives, tf.int32))

    # hard negative mining for classification
    predictions_for_bg = tf.nn.softmax(cls_pred)[:, 0]
    prob_for_negtives = tf.where(negtive_mask,
                           0. - predictions_for_bg,
                           # ignore all the positives
                           0. - tf.ones_like(predictions_for_bg))
    topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=n_neg_to_select)
    selected_neg_mask = prob_for_negtives > topk_prob_for_bg[-1]

    # # random select negtive examples for classification
    # selected_neg_mask = tf.random_uniform(tf.shape(gscores), minval=0, maxval=1.) < tf.where(
    #                                                                                     tf.greater(n_negtives, 0),
    #                                                                                     tf.divide(tf.cast(n_neg_to_select, tf.float32), n_negtives),
    #                                                                                     tf.zeros_like(tf.cast(n_neg_to_select, tf.float32)),
    #                                                                                     name='rand_select_negtive')

    # include both selected negtive and all positive examples
    final_mask = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, selected_neg_mask), positive_mask))
    total_examples = tf.reduce_sum(tf.cast(final_mask, tf.float32))

    # add mask for glabels and cls_pred here
    glabels = tf.boolean_mask(tf.clip_by_value(glabels, 0, FLAGS.num_classes), tf.stop_gradient(final_mask))
    cls_pred = tf.boolean_mask(cls_pred, tf.stop_gradient(final_mask))
    location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
    gtargets = tf.boolean_mask(gtargets, tf.stop_gradient(positive_mask))
    predictions = {
        'classes': tf.argmax(cls_pred, axis=-1),
        'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
        'bboxes_predict': tf.reshape(bboxes_pred, [-1, 4]) }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.cond(n_positives > 0., lambda: tf.losses.sparse_softmax_cross_entropy(labels=glabels, logits=cls_pred), lambda: 0.)
    #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=glabels, logits=cls_pred)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    loc_loss = tf.cond(n_positives > 0., lambda: modified_smooth_l1(location_pred, tf.stop_gradient(gtargets), sigma=1.), lambda: tf.zeros_like(location_pred))
    #loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1))
    loc_loss = tf.identity(loc_loss, name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = 1.2 * (cross_entropy + loc_loss) + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])
    total_loss = tf.identity(loss, name='total_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype))
        # Create a tensor named learning_rate for logging purposes.
        tf.identity(truncated_learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                                momentum=params['momentum'])

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    cls_accuracy = tf.metrics.accuracy(glabels, predictions['classes'])
    metrics = {'cls_accuracy': cls_accuracy}

    # Create a tensor named train_accuracy for logging purposes.
    tf.identity(cls_accuracy[1], name='cls_accuracy')
    tf.summary.scalar('cls_accuracy', cls_accuracy[1])

    return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=metrics,
          scaffold = tf.train.Scaffold(init_fn=train_helper.get_init_fn_for_scaffold(FLAGS)))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    xdetector = tf.estimator.Estimator(
        model_fn=xdet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
            'decay_steps': FLAGS.decay_steps,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })

    tensors_to_log = {
        'lr': 'learning_rate',
        'ce_loss': 'cross_entropy_loss',
        'loc_loss': 'location_loss',
        'total_loss': 'total_loss',
        'cls_acc': 'cls_accuracy',
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting a training cycle.')
    xdetector.train(input_fn=input_pipeline(), hooks=[logging_hook])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
