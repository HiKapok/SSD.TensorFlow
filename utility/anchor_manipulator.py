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
import math

import tensorflow as tf
import numpy as np

from tensorflow.contrib.image.python.ops import image_ops


def areas(gt_bboxes):
    with tf.name_scope('bboxes_areas', [gt_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        return (xmax - xmin + 1.) * (ymax - ymin + 1.)

def intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection', [gt_bboxes, default_bboxes]):
        # num_anchors x 1
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        # 1 x num_anchors
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        # broadcast here to generate the full matrix
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin + 1., 0.)
        w = tf.maximum(int_xmax - int_xmin + 1., 0.)

        return h * w
def iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', [gt_bboxes, default_bboxes]):
        inter_vol = intersection(gt_bboxes, default_bboxes)
        # broadcast
        areas_gt = areas(gt_bboxes)
        union_vol = areas_gt + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_vol

        #areas_gt = tf.Print(areas_gt, [areas_gt], summarize=100)
        return tf.where(tf.equal(union_vol, 0.0), tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''do_dual_max_match, but using the transpoed overlap matrix, this may be faster due to the cache friendly

    Args:
        overlap_matrix: num_anchors * num_gt
    '''
    with tf.name_scope('dual_max_match', [overlap_matrix]):
        # first match from anchors' side
        anchors_to_gt = tf.argmax(overlap_matrix, axis=1)
        # the matching degree
        match_values = tf.reduce_max(overlap_matrix, axis=1)

        #positive_mask = tf.greater(match_values, high_thres)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        negative_mask = less_mask if ignore_between else between_mask
        ignore_mask = between_mask if ignore_between else less_mask
        # comment following two lines
        # over_pos_mask = tf.greater(match_values, 0.7)
        # ignore_mask = tf.logical_or(ignore_mask, over_pos_mask)
        # fill all negative positions with -1, all ignore positions is -2
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

        # negtive values has no effect in tf.one_hot, that means all zeros along that axis
        # so all positive match positions in anchors_to_gt_mask is 1, all others are 0
        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[1], tf.int64)),
                                        tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        # match from ground truth's side
        gt_to_anchors = tf.argmax(overlap_matrix, axis=0)
        gt_to_anchors_overlap = tf.reduce_max(overlap_matrix, axis=0, keepdims=True)

        #gt_to_anchors = tf.Print(gt_to_anchors, [tf.equal(overlap_matrix, gt_to_anchors_overlap)], message='gt_to_anchors_indices:', summarize=100)
        # the max match from ground truth's side has higher priority
        left_gt_to_anchors_mask = tf.equal(overlap_matrix, gt_to_anchors_overlap)#tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[0], on_value=True, off_value=False, axis=0, dtype=tf.bool)
        if not gt_max_first:
            # the max match from anchors' side has higher priority
            # use match result from ground truth's side only when the the matching degree from anchors' side is lower than position threshold
            left_gt_to_anchors_mask = tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=0, keep_dims=True) < 1, left_gt_to_anchors_mask)
        # can not use left_gt_to_anchors_mask here, because there are many ground truthes match to one anchor, we should pick the highest one even when we are merging matching from ground truth side
        left_gt_to_anchors_mask = tf.to_int64(left_gt_to_anchors_mask)
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        # merge matching results from ground truth's side with the original matching results from anchors' side
        # then select all the overlap score of those matching pairs
        selected_scores = tf.gather_nd(overlap_matrix,  tf.stack([tf.range(tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                                                    tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=1) > 0,
                                                                            tf.argmax(left_gt_to_anchors_scores, axis=1),
                                                                            anchors_to_gt)], axis=1))
        # return the matching results for both foreground anchors and background anchors, also with overlap scores
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=1) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=1),
                        match_indices), selected_scores

# def save_anchors(bboxes, labels, anchors_point):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     np.save('./debug/bboxes_{}.npy'.format(save_image_with_bbox.counter), np.copy(bboxes))
#     np.save('./debug/labels_{}.npy'.format(save_image_with_bbox.counter), np.copy(labels))
#     np.save('./debug/anchors_{}.npy'.format(save_image_with_bbox.counter), np.copy(anchors_point))
#     return save_image_with_bbox.counter

class AnchorEncoder(object):
    def __init__(self, positive_threshold, ignore_threshold, prior_scaling):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling

    def center2point(self, center_y, center_x, height, width):
        with tf.name_scope('center2point'):
            return center_y - (height - 1.) / 2., center_x - (width - 1.) / 2., center_y + (height - 1.) / 2., center_x + (width - 1.) / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        with tf.name_scope('point2center'):
            height, width = (ymax - ymin + 1.), (xmax - xmin + 1.)
            return (ymin + ymax) / 2., (xmin + xmax) / 2., height, width

    def get_anchors_width_height(self, anchor_scale, extra_anchor_scale, anchor_ratio, name=None):
        '''get_anchors_width_height

        Given scales and ratios, generate anchors along depth (you should use absolute scale in the input image)
        Args:
            anchor_scale: base scale of the window size used to transform anchors, each scale should have every ratio in 'anchor_ratio'
            extra_anchor_scale: base scale of the window size used to transform anchors, each scale should have ratio of 1:1
            anchor_ratio: all ratios of anchors for each scale in 'anchor_scale'
        '''
        with tf.name_scope(name, 'get_anchors_width_height'):
            all_num_anchors_depth = len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale)

            list_h_on_image = []
            list_w_on_image = []

            # for square anchors
            for _, scale in enumerate(extra_anchor_scale):
                list_h_on_image.append(scale)
                list_w_on_image.append(scale)
            # for other aspect ratio anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    list_h_on_image.append(scale / math.sqrt(ratio))
                    list_w_on_image.append(scale * math.sqrt(ratio))
            # shape info:
            # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
            # h_on_image, w_on_image: num_anchors_along_depth
            return tf.constant(list_h_on_image, dtype=tf.float32), tf.constant(list_w_on_image, dtype=tf.float32), all_num_anchors_depth

    def generate_anchors_by_offset(self, anchors_height, anchors_width, anchor_depth, image_shape, layer_shape, feat_stride, offset=0.5, name=None):
        '''generate_anchors_by_offset

        Given anchor width and height, generate tiled anchors across the 'layer_shape'
        Args:
            anchors_height, anchors_width, anchor_depth: generate by the above function 'get_anchors_width_height'
            image_shape: the input image size, since we will generate anchors in absolute coordinates, [height, width]
            layer_shape: the size of layer on which we will tile the anchors, [height, width]
            feat_stride: the strides from input image to the layer on which we will generate anchors
            offset: the offset (height offset and width offset) in in the feature map when we tile anchors, should be either single scalar or a list of scalar
        '''
        with tf.name_scope(name, 'generate_anchors'):
            image_height, image_width, feat_stride = tf.to_float(image_shape[0]), tf.to_float(image_shape[1]), tf.to_float(feat_stride)

            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))

            if isinstance(offset, list):
                tf.logging.info('{}: Using seperate offset: height: {}, width: {}.'.format(name, offset[0], offset[1]))
                offset_h = offset[0]
                offset_w = offset[1]
            else:
                offset_h = offset
                offset_w = offset
            y_on_image = (tf.to_float(y_on_layer) + offset_h) * feat_stride
            x_on_image = (tf.to_float(x_on_layer) + offset_w) * feat_stride

            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax = self.center2point(tf.expand_dims(y_on_image, axis=-1),
                                                                                    tf.expand_dims(x_on_image, axis=-1),
                                                                                    anchors_height, anchors_width)

            anchors_ymin = tf.reshape(anchors_ymin, [-1, anchor_depth])
            anchors_xmin = tf.reshape(anchors_xmin, [-1, anchor_depth])
            anchors_ymax = tf.reshape(anchors_ymax, [-1, anchor_depth])
            anchors_xmax = tf.reshape(anchors_xmax, [-1, anchor_depth])

            return anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax

    def get_anchors_count(self, anchors_depth, layer_shape, name=None):
        '''get_anchors_count

        Return the total anchors on specific layer
        Args:
            anchor_depth: generate by the above function 'get_anchors_width_height'
            layer_shape: the size of layer on which we will tile the anchors, [height, width]
        '''
        with tf.name_scope(name, 'get_anchors_count'):
            all_num_anchors_spatial = layer_shape[0] * layer_shape[1]
            all_num_anchors = all_num_anchors_spatial * anchors_depth
            return all_num_anchors_spatial, all_num_anchors

    def get_all_anchors(self, image_shape, anchors_height, anchors_width, anchors_depth, anchors_offsets, layer_shapes, feat_strides, allowed_borders, should_clips, name=None):
        '''get_all_anchors

        Return the all anchors from all layers
        Args:
            image_shape: the input image size, since we will generate anchors in absolute coordinates, [height, width]
            anchors_height: list, each of which is generated by the above function 'get_anchors_width_height'
            anchors_width: list, each of which is generated by the above function 'get_anchors_width_height'
            anchors_depth: list, each of which is generated by the above function 'get_anchors_width_height'
            anchors_offsets: list, each of which will be used by 'generate_anchors_by_offset'
            layer_shapes: list, each of which will be used by 'generate_anchors_by_offset'
            feat_strides: list, each of which will be used by 'generate_anchors_by_offset'
            allowed_borders: list, each of which is the border margin to clip border anchors for each layer
            should_clips: list, each of which indicate that if we should clip anchors to image border for each layer
        '''
        with tf.name_scope(name, 'get_all_anchors'):
            image_height, image_width = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])

            anchors_ymin = []
            anchors_xmin = []
            anchors_ymax = []
            anchors_xmax = []
            anchor_allowed_borders = []
            for ind, anchor_depth in enumerate(anchors_depth):
                with tf.name_scope('generate_anchors_{}'.format(ind)):
                    _anchors_ymin, _anchors_xmin, _anchors_ymax, _anchors_xmax = self.generate_anchors_by_offset(anchors_height[ind], anchors_width[ind], anchor_depth, image_shape, layer_shapes[ind], feat_strides[ind], offset=anchors_offsets[ind])

                    if should_clips[ind]:
                        _anchors_ymin = tf.clip_by_value(_anchors_ymin, 0., image_height - 1.)
                        _anchors_xmin = tf.clip_by_value(_anchors_xmin, 0., image_width - 1.)
                        _anchors_ymax = tf.clip_by_value(_anchors_ymax, 0., image_height - 1.)
                        _anchors_xmax = tf.clip_by_value(_anchors_xmax, 0., image_width - 1.)

                    _anchors_ymin = tf.reshape(_anchors_ymin, [-1])
                    _anchors_xmin = tf.reshape(_anchors_xmin, [-1])
                    _anchors_ymax = tf.reshape(_anchors_ymax, [-1])
                    _anchors_xmax = tf.reshape(_anchors_xmax, [-1])

                    anchors_ymin.append(_anchors_ymin)
                    anchors_xmin.append(_anchors_xmin)
                    anchors_ymax.append(_anchors_ymax)
                    anchors_xmax.append(_anchors_xmax)
                    anchor_allowed_borders.append(tf.ones_like(_anchors_ymin, dtype=tf.float32) * allowed_borders[ind])

            # anchors_ymin = tf.reshape(tf.concat(anchors_ymin, axis=0), [-1])
            # anchors_xmin = tf.reshape(tf.concat(anchors_xmin, axis=0), [-1])
            # anchors_ymax = tf.reshape(tf.concat(anchors_ymax, axis=0), [-1])
            # anchors_xmax = tf.reshape(tf.concat(anchors_xmax, axis=0), [-1])
            # anchor_allowed_borders = tf.reshape(tf.concat(anchor_allowed_borders, axis=0), [-1])
            anchors_ymin = tf.concat(anchors_ymin, axis=0)
            anchors_xmin = tf.concat(anchors_xmin, axis=0)
            anchors_ymax = tf.concat(anchors_ymax, axis=0)
            anchors_xmax = tf.concat(anchors_xmax, axis=0)
            anchor_allowed_borders = tf.concat(anchor_allowed_borders, axis=0)

            inside_mask = tf.logical_and(tf.logical_and(anchors_ymin > -anchor_allowed_borders,
                                                        anchors_xmin > -anchor_allowed_borders),
                                        tf.logical_and(anchors_ymax < (image_height - 1. + anchor_allowed_borders),
                                                        anchors_xmax < (image_width - 1. + anchor_allowed_borders)))

            return anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask

    def encode_anchors(self, labels, bboxes, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask, debug=False):
        '''encode anchors with ground truth on the fly

        We generate prediction targets for all locations of the rpn feature map, so this routine is called when the final rpn feature map has been generated, so there is a performance bottleneck here but we have no idea to fix this because of we must perform multi-scale training. Maybe this needs to be placed on CPU, leave this problem to later

        Args:
            bboxes: [num_bboxes, 4] in [ymin, xmin, ymax, xmax] format
            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax, inside_mask: generate by 'get_all_anchors'
        '''
        with tf.name_scope('encode_anchors'):
            all_anchors = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)
            overlap_matrix = iou_matrix(all_anchors, bboxes) * tf.cast(tf.expand_dims(inside_mask, 1), tf.float32)
            #overlap_matrix = tf.Print(overlap_matrix, [tf.shape(overlap_matrix)])
            #matched_gt, gt_scores = custom_op.small_mining_match(overlap_matrix, 0., self._rpn_negtive_threshold, self._rpn_positive_threshold, 5, 0.35)
            matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
            # get all positive matching positions
            matched_gt_mask = matched_gt > -1
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)

            gt_labels = tf.gather(labels, matched_indices)
            # filter the invalid labels
            gt_labels = gt_labels * tf.to_int64(matched_gt_mask)
            # set those ignored positions to -1
            gt_labels = gt_labels + (-1 * tf.to_int64(matched_gt < -1))

            gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(tf.gather(bboxes, matched_indices), 4, axis=-1)

            # transform to center / size.
            gt_cy, gt_cx, gt_h, gt_w = self.point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            # encode features.
            # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[0]
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[1]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[2]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[3]
            # now gt_localizations is our regression object, but also maybe chaos at those non-positive positions
            if debug:
                gt_targets = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)
            else:
                gt_targets = tf.stack([gt_cy, gt_cx, gt_h, gt_w], axis=-1)
            # set all targets of non-positive positions to 0
            gt_targets = tf.expand_dims(tf.to_float(matched_gt_mask), -1) * gt_targets

            return gt_targets, gt_labels, gt_scores

    def batch_decode_anchors(self, pred_location, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax):
        '''batch_decode_anchors

        Args:
            pred_location: [batch, num_preds, 4] in yxhw format
            all_anchors: generate by 'get_all_anchors'
        '''
        with tf.name_scope('decode_rpn', [pred_location, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax]):
            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax = tf.expand_dims(anchors_ymin, axis=0), \
                                                                    tf.expand_dims(anchors_xmin, axis=0),\
                                                                    tf.expand_dims(anchors_ymax, axis=0),\
                                                                    tf.expand_dims(anchors_xmax, axis=0)
            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            pred_h = tf.exp(pred_location[:, :, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, :, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, :, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, :, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1)
    def decode_anchors(self, pred_location, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax):
        '''decode_anchors

        Args:
            pred_location: [num_preds, 4] in yxhw format
            all_anchors: generate by 'get_all_anchors'
        '''
        with tf.name_scope('decode_rpn', [pred_location, anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax]):
            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            pred_h = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1)
