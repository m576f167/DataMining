#!/usr/bin/python3
"""
/**
 ******************************************************************************
 * @file yolov3.py
 * @author Mohammad Isyroqi Fathan
 * $Rev: 1 $
 * $Date: Sat Sep 15 19:13:26 CDT 2018 $
 * @brief Defines the yolov3 class.
 ******************************************************************************
 * @copyright
 * @internal
 *
 * @endinternal
 *
 * @details
 * This file contains the class implementation of yolov3 model.
 * It is implemented based on https://github.com/mystic123/tensorflow-yolo-v3
 * and https://github.com/ztq0307/yolo-v3-tf
 *
 * @ingroup model
 */
"""

import tensorflow as tf
import numpy as np

class Model:
    """
    Class of yolov3  model.

    This class is the implementation of yolov3 model.
    """
    def __init__(self, params = None):
        # Initialization Code
        self.train_op = None
        self.update_metrics = None
        self.loss = None
        self.summary_op = None
        self.summary_sample_op = None
        self.global_step = None
        self.metrics = None
        self.metrics_variables = None
        self.metrics_init_op = None
        self.params = params
        self.inputs = None
        self.gt = None
        self.gt_mask = None
        self.inputs_height = 0
        self.inputs_width = 0
        self.inputs_channels = 3
        self.assign_ops = []
        self.metrics_mean_raw = {}
        self.__model_name = "Yolov3"
        self.__optimizer = None
        self.__out = None
        self.__predictions = None
        self.__predictions_boxes = None
        self.__draw_predictions = None
        self.__placeholder_predictions = None
        self.__placeholder_images = None
        self.placeholder_sample = []
        self.__summary_sample = []
        self.__summary_list = []
        self.__number_sample = 0
        self.__update_sample = None
        self.__weights_file = ''

        # Adam Optimizer Related
        self.adam_learning_rate = params['adam_learning_rate'] if self.__checkParamKey(params, "adam_learning_rate") else 1e-5
        self.adam_beta_1 = params['adam_beta_1'] if self.__checkParamKey(params, "adam_beta_1") else 0.9
        self.adam_beta_2 = params['adam_beta_2'] if self.__checkParamKey(params, "adam_beta_2") else 0.999
        self.adam_epsilon = params['adam_epsilon'] if self.__checkParamKey(params, "adam_epsilon") else 1e-08

        # Yolov3 related
        self.__grid_sizes = []
        self.scale_class = params['scale_class'] if self.__checkParamKey(params, 'scale_class') else 1.0
        self.scale_object = params['scale_object'] if self.__checkParamKey(params, 'scale_object') else 5.0
        self.scale_noobject = params['scale_noobject'] if self.__checkParamKey(params, 'scale_noobject') else 0.5
        self.scale_coord = params['scale_coord'] if self.__checkParamKey(params, 'scale_coord') else 1.0
        self.iou_threshold = params['iou_threshold'] if self.__checkParamKey(params, 'iou_threshold') else 0.4
        self.confidence_threshold = params['confidence_threshold'] if self.__checkParamKey(params, 'confidence_threshold') else 0.5
        self.bn_epsilon = params['bn_epsilon'] if self.__checkParamKey(params, 'bn_epsilon') else 1e-05
        self.bn_momentum = params['bn_momentum'] if self.__checkParamKey(params, 'bn_momentum') else 0.9
        self.leaky_relu_alpha = params['leaky_relu_alpha'] if self.__checkParamKey(params, 'leaky_relu_alpha') else 0.1
        self.num_classes = params['num_classes'] if self.__checkParamKey(params, 'num_classes') else 1
        self.anchors = params['anchors'] if self.__checkParamKey(params, 'anchors') else [[(116, 90), (156, 198), (373, 326)],
                                                                                            [(30, 61), (62, 45), (59, 119)],
                                                                                            [(10, 13), (16, 30), (33, 23)]]
        self.is_training = params['training'] if self.__checkParamKey(params, 'training') else True

    def __checkParamKey(self, params, key):
        return((isinstance(params, dict)) and (not (params.get(key) == None)))

    def buildModel(self):
        with tf.variable_scope(self.__model_name, reuse = tf.AUTO_REUSE):
            self.__out = self.__yoloV3(self.inputs, self.num_classes)
            out = self.__out
        self.__init_predict()
        self.__Loss()
        self.__training()
        self.__metricsAndSummary()
        return(out, self.__predictions_boxes)

    def getOut(self):
        return(self.__out)

    def getPredictions(self):
        return(self.__predictions)

    def getPredictionsBoxes(self):
        return(self.__predictions_boxes)

    def getDrawPredictions(self):
        return(self.__draw_predictions)

    def __buildDarknet53(self, inputs):
        inputs = self.__conv2dFixedPadding(inputs, 32, 3)
        inputs = self.__conv2dFixedPadding(inputs, 64, 3, strides = 2)
        inputs = self.__darknet53Block(inputs, 32)
        inputs = self.__conv2dFixedPadding(inputs, 128, 3, strides = 2)

        for i in range(2):
            inputs = self.__darknet53Block(inputs, 64)

        inputs = self.__conv2dFixedPadding(inputs, 256, 3, strides = 2)

        for i in range(8):
            inputs = self.__darknet53Block(inputs, 128)

        route_1 = inputs
        inputs = self.__conv2dFixedPadding(inputs, 512, 3, strides = 2)

        for i in range(8):
            inputs = self.__darknet53Block(inputs, 256)

        route_2 = inputs
        inputs = self.__conv2dFixedPadding(inputs, 1024, 3, strides = 2)

        for i in range(4):
            inputs = self.__darknet53Block(inputs, 512)

        return route_1, route_2, inputs

    def __darknet53Block(self, inputs, filters):
        shortcut = inputs
        inputs = self.__conv2dFixedPadding(inputs, filters, 1)
        inputs = self.__conv2dFixedPadding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def __conv2dFixedPadding(self, inputs, filters, kernel_size, strides = 1):
        if strides > 1:
            inputs = self.__fixedPadding(inputs, kernel_size)
        inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides = strides,
                                  padding = ("SAME" if strides == 1 else "VALID"),
                                  data_format = "channels_first",
                                  use_bias = False)
        inputs = tf.layers.batch_normalization(inputs, epsilon = self.bn_epsilon,
                                               axis = 1,
                                               momentum = self.bn_momentum,
                                               training = self.is_training)
        inputs = tf.nn.leaky_relu(inputs, alpha = self.leaky_relu_alpha)
        return inputs

    def __fixedPadding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(inputs, [[0, 0],
                                        [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]], mode = "CONSTANT")
        return padded_inputs

    def __yoloBlock(self, inputs, filters):
        inputs = self.__conv2dFixedPadding(inputs, filters, 1)
        inputs = self.__conv2dFixedPadding(inputs, filters * 2, 3)
        inputs = self.__conv2dFixedPadding(inputs, filters, 1)
        inputs = self.__conv2dFixedPadding(inputs, filters * 2, 3)
        inputs = self.__conv2dFixedPadding(inputs, filters, 1)
        route = inputs
        inputs = self.__conv2dFixedPadding(inputs, filters * 2, 3)
        return route, inputs

    def __detectionLayer(self, inputs, num_classes, anchors):
        num_anchors = len(anchors)
        predictions = tf.layers.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                                       data_format = "channels_first",
                                       strides = 1)
        shape = predictions.get_shape().as_list()
        grid_size = shape[2:4]
        self.__grid_sizes.append((grid_size[0], grid_size[1]))
        dim = grid_size[0] * grid_size[1]
        bbox_attrs = 5 + num_classes

        predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])
        predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
        stride = (self.inputs_height // grid_size[0], self.inputs_width // grid_size[1])
        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

        box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis = -1)

        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[1], dtype = tf.float32)
        grid_y = tf.range(grid_size[0], dtype = tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([y_offset, x_offset], axis = -1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride

        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * anchors
        box_sizes = tf.maximum(box_sizes * stride, 1)
        box_sizes = tf.minimum(box_sizes, 1e+17)

        detections = tf.concat([box_centers, box_sizes, confidence], axis = -1)

        classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis = -1)
        predictions = tf.reshape(predictions, [-1, grid_size[0], grid_size[1], num_anchors,
                                               bbox_attrs])
        return predictions

    def __upsample(self, inputs, out_shape):
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[2]
        new_width = out_shape[3]

        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

        inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = tf.identity(inputs, name = "upsampled")
        return inputs

    def __yoloV3(self, inputs, num_classes):
        with tf.variable_scope("Normalize_And_Transpose"):
            inputs = inputs / 255
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        with tf.variable_scope("Backbone"):
            route_1, route_2, inputs = self.__buildDarknet53(inputs)

        with tf.variable_scope("Yolo_layer"):
            route, inputs = self.__yoloBlock(inputs, 512)
            detect_1 = self.__detectionLayer(inputs, self.num_classes, self.anchors[0])
            detect_1 = tf.identity(detect_1, name = 'detect_1')

            inputs = self.__conv2dFixedPadding(route, 256, 1)
            upsample_size = route_2.get_shape().as_list()
            inputs = self.__upsample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_2], axis = 1)

            route, inputs = self.__yoloBlock(inputs, 256)

            detect_2 = self.__detectionLayer(inputs, self.num_classes, self.anchors[1])
            detect_2 = tf.identity(detect_2, name = 'detect_2')

            inputs = self.__conv2dFixedPadding(route, 128, 1)
            upsample_size = route_1.get_shape().as_list()
            inputs = self.__upsample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_1], axis = 1)

            _, inputs = self.__yoloBlock(inputs, 128)

            detect_3 = self.__detectionLayer(inputs, self.num_classes, self.anchors[2])
            detect_3 = tf.identity(detect_3, name = "detect_3")

            # detections = tf.concat([detect_1, detect_2, detect_3], axis = 1)
            # detections = tf.identity(detections, name = 'detections')
            detections = [detect_1, detect_2, detect_3]
            return detections

    def __reshapePredictions(self):
        predictions = []
        for i in range(len(self.__grid_sizes)):
            prediction = self.__out[i]
            num_anchors = len(self.anchors[i])
            grid_size = self.__grid_sizes[i]
            dim = grid_size[0] * grid_size[1]
            bbox_attrs = 5 + self.num_classes
            prediction = tf.reshape(prediction, [-1, dim * num_anchors, bbox_attrs])
            predictions.append(prediction)
        predictions = tf.concat(predictions, axis = 1)
        predictions = tf.identity(predictions, name = "predictions")
        self.__predictions = predictions
        return(self.__predictions)

    def __predictionBoxes(self):
        center_y, center_x, height, width, attrs = tf.split(self.__predictions, [1, 1, 1, 1, -1], axis = -1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2

        boxes = tf.concat([y0, x0, y1, x1], axis = -1)
        self.__predictions_boxes = tf.concat([boxes, attrs], axis = -1)
        return self.__predictions_boxes

    def __init_predict(self):
        with tf.variable_scope("predictions", reuse = tf.AUTO_REUSE):
            self.__reshapePredictions()
            self.__predictionBoxes()
        with tf.variable_scope("predict", reuse = tf.AUTO_REUSE):
            self.__placeholder_predictions = tf.placeholder(tf.float32)
            self.__placeholder_images = tf.placeholder(tf.float32)
            results = tf.image.draw_bounding_boxes(self.__placeholder_images,
                                                   self.__placeholder_predictions)
            self.__draw_predictions = results
            self.placeholder_sample = []
            self.__summary_sample = []
            for i in range(self.__number_sample):
                self.placeholder_sample.append(tf.placeholder(tf.float32))
                self.__summary_sample.append(tf.summary.image("Sample_" + str(i), self.placeholder_sample[i]))
            self.summary_sample_op = tf.summary.merge(self.__summary_sample)

    def __iouBoxes(self, box1, box2):
        b1_y0, b1_x0, b1_y1, b1_x1 = box1
        b2_y0, b2_x0, b2_y1, b2_x1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou

    def nonMaxSuppression(self, predictions_boxes):
        conf_mask = np.expand_dims((predictions_boxes[:, :, 4] > self.confidence_threshold), -1)
        predictions = predictions_boxes * conf_mask

        results = []
        for i, image_pred, in enumerate(predictions):
            result = {}
            shape = image_pred.shape
            non_zero_idxs = np.nonzero(image_pred)
            image_pred = image_pred[non_zero_idxs]
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis = -1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                cls_scores = cls_boxes[:, -1]
                cls_boxes = cls_boxes[:, :-1]

                while len(cls_boxes) > 0:
                    box = cls_boxes[0]
                    score = cls_scores[0]
                    if not cls in result:
                        result[cls] = []
                    result[cls].append((box, score))
                    cls_boxes = cls_boxes[1:]
                    ious = np.array([self.__iouBoxes(box, x) for x in cls_boxes])
                    iou_mask = ious < self.iou_threshold
                    cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]
            results.append(result)
        return results

    def processPredictions(self, sess, predictions, input_image, gt, gt_mask):
        nms_result = self.nonMaxSuppression(predictions)
        # TODO: Support multiple objects and classes in 1 image
        # gt_bbox = np.array(list(map(lambda x: np.concatenate([np.min(np.where(x > 0), axis = 1)[0:2],
        #                                                       np.max(np.where(x > 0), axis = 1)[0:2]]), gt)))
        image_result = input_image + (gt_mask * 20/100)
        result = []
        for i, image in enumerate(image_result):
            pred = nms_result[i]
            pred_image = image
            for class_assignment in pred.keys():
                bboxes = [x/[self.inputs_height, self.inputs_width, self.inputs_height, self.inputs_width] for x, score in pred[class_assignment]]
                scores = [x for bbox, x in pred[class_assignment]]
                pred_image = sess.run(self.__draw_predictions, feed_dict={self.__placeholder_predictions : [bboxes],
                                                                          self.__placeholder_images : [pred_image]})
            result.append(pred_image)
        result = np.array(result)
        return(result)

    def __centerResponseTrue(self, gt_confidence, gt_box_centers, stride):
        x_center = tf.cast(tf.floor(gt_box_centers[0, 0, 0, 1] / stride[1]), dtype = tf.int32)
        y_center = tf.cast(tf.floor(gt_box_centers[0, 0, 0, 0] / stride[0]), dtype = tf.int32)

        shape_confidence = gt_confidence.get_shape().as_list()
        mask_confidence = tf.ones([1, 1, shape_confidence[2], 1], dtype = tf.float32)
        gt_confidence = tf.pad(mask_confidence, [[y_center, shape_confidence[0] - y_center - 1],
                                                 [x_center, shape_confidence[1] - x_center - 1],
                                                 [0, 0],
                                                 [0, 0]], "CONSTANT")
        return gt_confidence

    def __centerResponse(self, gt, num_classes, stride):
        gt_box_centers, gt_box_sizes, gt_confidence, gt_classes = tf.split(gt, [2, 2, 1, num_classes], axis = -1)

        gt_confidence = tf.cond(tf.count_nonzero(gt_confidence) > 0,
                                lambda: self.__centerResponseTrue(gt_confidence, gt_box_centers, stride),
                                lambda: gt_confidence)
        gt = tf.concat([gt_box_centers, gt_box_sizes, gt_confidence, gt_classes], axis = -1)
        return gt

    def __preprocessGT(self, gt, num_classes, anchors):
        gt_processed = []
        for i in range(len(self.__grid_sizes)):
            num_anchors = len(anchors[i])
            dim = self.__grid_sizes[i][0] * self.__grid_sizes[i][1]
            bbox_attrs = 5 + num_classes

            stride = (self.inputs_height // self.__grid_sizes[i][0], self.inputs_width // self.__grid_sizes[i][1])

            gt_bbox = tf.reshape(gt, [-1, 1, 1, 1, bbox_attrs])
            gt_bbox = tf.tile(gt_bbox, [1, self.__grid_sizes[i][0], self.__grid_sizes[i][1], num_anchors, 1])
            gt_bbox = tf.map_fn(lambda x: self.__centerResponse(x, num_classes, stride), gt_bbox)
            # gt_bbox = tf.reshape(gt_bbox, [-1, num_anchors * dim, bbox_attrs])

            gt_processed.append(gt_bbox)

        # predictions = tf.concat(gt_processed, axis = 1)
        return gt_processed

    def __lossIou(self, predictions, gt):
        xmin1 = predictions[:, :, :, :, 1:2] - predictions[:, :, :, :, 3:4]/2
        xmax1 = predictions[:, :, :, :, 1:2] + predictions[:, :, :, :, 3:4]/2
        ymin1 = predictions[:, :, :, :, 0:1] - predictions[:, :, :, :, 2:3]/2
        ymax1 = predictions[:, :, :, :, 0:1] + predictions[:, :, :, :, 2:3]/2
        xmin2 = gt[:, :, :, :, 1:2] - gt[:, :, :, :, 3:4]/2
        xmax2 = gt[:, :, :, :, 1:2] + gt[:, :, :, :, 3:4]/2
        ymin2 = gt[:, :, :, :, 0:1] - gt[:, :, :, :, 2:3]/2
        ymax2 = gt[:, :, :, :, 0:1] + gt[:, :, :, :, 2:3]/2

        box1 = tf.stack([ymin1, xmin1, ymax1, xmax1], axis = -1)
        box2 = tf.stack([ymin2, xmin2, ymax2, xmax2], axis = -1)

        inter_h = tf.minimum(box1[:, :, :, :, :, 2], box2[:, :, :, :, :, 2]) - tf.maximum(box1[:, :, :, :, :, 0],
                                                                                          box2[:, :, :, :, :, 0])
        inter_w = tf.minimum(box1[:, :, :, :, :, 3], box2[:, :, :, :, :, 3]) - tf.maximum(box1[:, :, :, :, :, 1],
                                                                                          box2[:, :, :, :, :, 1])

        intersection = inter_w * inter_h

        square_1 = predictions[:, :, :, :, 2:3] * predictions[:, :, :, :, 3:4]
        square_2 = gt[:, :, :, :, 2:3] * gt[:, :, :, :, 3:4]

        union = square_1 + square_2 - intersection

        return tf.clip_by_value(intersection / union, 0, 1)

    def __Loss(self):
        with tf.device('/cpu:0'), tf.variable_scope("preprocess_gt", reuse = tf.AUTO_REUSE):
            preprocessed_gt = self.__preprocessGT(self.gt, self.num_classes, self.anchors)
        with tf.variable_scope("loss_function", reuse = tf.AUTO_REUSE):
            self.loss = 0
            for i in range(len(self.__grid_sizes)):
                shape = tf.shape(self.__out[i])
                batch_size = tf.cast(shape[0], dtype = tf.float32)
                box_centers, box_sizes, confidence, classes = tf.split(self.__out[i],
                                                                    [2, 2, 1, self.num_classes],
                                                                    axis = -1)
                gt_box_centers, gt_box_sizes, gt_confidence, gt_classes = tf.split(preprocessed_gt[i],
                                                                                [2, 2, 1, self.num_classes],
                                                                                axis = -1)
                iou = self.__lossIou(self.__out[i], preprocessed_gt[i])
                max_iou = tf.reduce_max(iou, axis = 3, keepdims = True)
                # max_iou_mask = tf.cast(tf.equal(iou, max_iou), tf.float32)
                iou_mask = tf.cast(max_iou >= self.iou_threshold, tf.float32) * gt_confidence # * max_iou_mask

                noobject_iou = (tf.ones_like(iou_mask) - iou_mask) * iou
                noobj_mask = tf.cast(noobject_iou < self.iou_threshold, tf.float32)

                # class_loss = self.scale_class * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = classes,
                #                                                                                       labels = gt_classes,
                #                                                                                       dim = 4) * gt_confidence) / batch_size
                loss_class = self.scale_class * tf.reduce_sum(tf.square((classes * gt_confidence) - (gt_classes * gt_confidence))) / batch_size
                self.__summary_list.append(tf.summary.scalar('loss_class', loss_class))
                self.metrics_mean_raw["loss_class_" + str(i)] = loss_class

                loss_obj = self.scale_object * tf.reduce_sum(tf.square((confidence * iou_mask) - iou_mask)) / batch_size
                self.__summary_list.append(tf.summary.scalar('loss_obj', loss_obj))
                self.metrics_mean_raw["loss_obj_" + str(i)] = loss_obj

                loss_noobj = self.scale_noobject * tf.reduce_sum(tf.square((confidence * noobj_mask) - 0)) / batch_size
                self.__summary_list.append(tf.summary.scalar('loss_noobj', loss_noobj))
                self.metrics_mean_raw["loss_noobj_" + str(i)] = loss_noobj

                loss_x = tf.reduce_sum(tf.square((box_centers[:, :, :, :, 1:2] * gt_confidence) - (gt_box_centers[:, :, :, :, 1:2] * gt_confidence))) / batch_size
                loss_y = tf.reduce_sum(tf.square((box_centers[:, :, :, :, 0:1] * gt_confidence) - (gt_box_centers[:, :, :, :, 0:1] * gt_confidence))) / batch_size
                # loss_w = tf.reduce_sum(tf.square((tf.sqrt(box_sizes[:, :, :, :, 1:2]) * gt_confidence) - (tf.sqrt(gt_box_sizes[:, :, :, :, 1:2]) * gt_confidence))) / batch_size
                # loss_h = tf.reduce_sum(tf.square((tf.sqrt(box_sizes[:, :, :, :, 0:1]) * gt_confidence) - (tf.sqrt(gt_box_sizes[:, :, :, :, 0:1]) * gt_confidence))) / batch_size
                loss_w = tf.reduce_sum(tf.square((box_sizes[:, :, :, :, 1:2] * gt_confidence) - (gt_box_sizes[:, :, :, :, 1:2] * gt_confidence))) / batch_size
                loss_h = tf.reduce_sum(tf.square((box_sizes[:, :, :, :, 0:1] * gt_confidence) - (gt_box_sizes[:, :, :, :, 0:1] * gt_confidence))) / batch_size
                self.__summary_list.append(tf.summary.scalar('loss_x', loss_x))
                self.__summary_list.append(tf.summary.scalar('loss_y', loss_y))
                self.__summary_list.append(tf.summary.scalar('loss_w', loss_w))
                self.__summary_list.append(tf.summary.scalar('loss_h', loss_h))
                self.metrics_mean_raw["loss_x_" + str(i)] = loss_x
                self.metrics_mean_raw["loss_y_" + str(i)] = loss_y
                self.metrics_mean_raw["loss_w_" + str(i)] = loss_w
                self.metrics_mean_raw["loss_h_" + str(i)] = loss_h

                loss_coord = self.scale_coord * tf.add_n([loss_x, loss_y, loss_w, loss_h])
                self.__summary_list.append(tf.summary.scalar('loss_coord', loss_coord))
                self.metrics_mean_raw["loss_coord_" + str(i)] = loss_coord

                self.loss += tf.add_n([loss_class, loss_obj, loss_noobj, loss_coord])
            self.__summary_list.append(tf.summary.scalar('loss', self.loss))
        # accuracy = tf.reduce_mean()

    def __training(self):
        self.__optimizer = tf.train.AdamOptimizer(learning_rate = self.adam_learning_rate,
                                                  beta1 = self.adam_beta_1,
                                                  beta2 = self.adam_beta_2,
                                                  epsilon = self.adam_epsilon)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = self.__optimizer.minimize(self.loss, global_step = self.global_step)

    def __metricsAndSummary(self):
        var_list = tf.global_variables(scope = self.__model_name)
        for var in var_list:
            if (("conv2d" in var.name.split('/')[-2]) or
                ("batch_normalization" in var.name.split('/')[-2])):
                self.__summary_list.append(tf.summary.histogram(var.name, var))
        # 'accuracy' : tf.metrics.accuracy()
        with tf.variable_scope("metrics", reuse = tf.AUTO_REUSE):
            self.metrics = {
                'loss_value' : tf.metrics.mean(self.loss),
                'score' : tf.metrics.mean(self.loss)
            }
            for metric in self.metrics_mean_raw.keys():
                self.metrics[metric] = tf.metrics.mean(self.metrics_mean_raw[metric])
            self.update_metrics = tf.group(*[op for _, op in self.metrics.values()])
            self.metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "metrics")
            self.metrics_init_op = tf.variables_initializer(self.metrics_variables)

        # Summary
        # tf.summary.scalar('accuracy', accuracy)
        # tf.summary.image()
        # self.summary_op = tf.summary.merge_all()
        self.summary_op = tf.summary.merge(self.__summary_list)

    def setInput(self, inputs, gt, gt_mask, height, width, channels, number_sample = 0):
        self.inputs = inputs
        self.gt = gt
        self.gt_mask = gt_mask
        self.inputs_height = height
        self.inputs_width = width
        self.inputs_channels = channels
        self.__number_sample = number_sample

    def loadWeights(self, weights_file):
        with tf.variable_scope("Load_Weights", reuse = tf.AUTO_REUSE):
            if weights_file == None:
                self.assign_ops = []
                return self.assign_ops
            print("[+] Loading weights file: ", weights_file)
            self.__weights_file = weights_file
            with open(weights_file, "rb") as fp:
                _ = np.fromfile(fp, dtype = np.int32, count = 5)
                weights = np.fromfile(fp, dtype = np.float32)
            ptr = 0
            i = 0
            assign_ops = []
            var_list = tf.global_variables(scope = self.__model_name)
            while i < len(var_list) - 1:
                var1 = var_list[i]
                var2 = var_list[i + 1]
                # do something only if we process conv layer
                if "conv2d" in var1.name.split('/')[-2]:
                    if "batch_normalization" in var2.name.split('/')[-2]:
                        # load batch norm params
                        gamma, beta, mean, var = var_list[i + 1:i + 5]
                        batch_norm_vars = [beta, gamma, mean, var]
                        for var in batch_norm_vars:
                            shape = var.shape.as_list()
                            num_params = np.prod(shape)
                            var_weights = weights[ptr:ptr + num_params].reshape(shape)
                            ptr += num_params
                            assign_ops.append(tf.assign(var, var_weights, validate_shape = True))
                        # Move pointer by 4, because loaded 4 variables
                        i += 4
                    elif "conv2d" in var2.name.split('/')[-2]:
                        # load biases
                        bias = var2
                        bias_shape = bias.shape.as_list()
                        bias_params = np.prod(bias_shape)
                        bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                        ptr += bias_params
                        assign_ops.append(tf.assign(bias, bias_weights, validate_shape = True))
                        # Loaded 1 variable
                        i += 1
                    # Load conv2d weights
                    shape = var1.shape.as_list()
                    num_params = np.prod(shape)

                    var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                    # transpose to column-major
                    var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                    ptr += num_params
                    assign_ops.append(tf.assign(var1, var_weights, validate_shape = True))
                    i += 1
                else:
                    # Not a conv layer
                    i += 1
            self.assign_ops = assign_ops
            print("[+] Finished loading weights file: ", weights_file)
            return self.assign_ops

