import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        # 卷积层，kernel的w，kernel的h，kernel的个数，步长w，步长h，名字（scpoe）
        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        # todo LSTM还得看看
        (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))
        # 边框点的预测，每个边框的高度点
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        # 边框评分，每个边框是否是背景
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # (self.feed('rpn_cls_score')
        #  .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
        #  .spatial_softmax(name='rpn_cls_prob'))
        #
        # # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        # (self.feed('rpn_cls_prob')
        #  .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_score', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))
