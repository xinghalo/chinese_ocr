import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from ctpn.lib.utils.timer import Timer
from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.test import  test_ctpn
from ctpn.lib.networks.factory import get_network
from ctpn.lib.text_connector.detectors import TextDetector
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    """
    调整图片尺寸

    :param im:
    :param scale:
    :param max_scale:
    :return:
    """
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    # 基于opencv调整图片尺寸，使用的方法是inter_linear
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def load_tf_model():
    # load config file
    cfg.TEST.checkpoints_path = './ctpn/checkpoints'

    # init session todo 不知道这些是什么意思
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)

    # load network
    net = get_network("VGGnet_test")

    # load model
    print('Loading network {:s}... '.format("VGGnet_test"))
    # todo 不懂这里是什么意思
    saver = tf.train.Saver()
    try:
        # todo tf的检查点
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    return sess, net

# 加载模型
sess, net = load_tf_model()

def ctpn(img):
    timer = Timer()
    timer.tic()

    # 调整图片尺寸，长和宽缩小到最小维度为900-1500
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # 获得分数和boxes信息
    scores, boxes = test_ctpn(sess, net, img)

    # 清理文本框
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    timer.toc()
    print("\n----------------------------------------------")
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return scores, boxes, img, scale

def draw_boxes(img, boxes, scale):
    box_id = 0
    img = img.copy()
    text_recs = np.zeros((len(boxes), 8), np.int)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        if box[8] >= 0.8:
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        for i in range(8):
            text_recs[box_id, i] = box[i]

        box_id += 1

    img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    return text_recs, img

def text_detect(img):
    # 基于CTPN生成检测信息
    scores, boxes, img, scale = ctpn(img)
    text_recs, img_drawed = draw_boxes(img, boxes, scale)
    return text_recs, img_drawed, img

if __name__ == '__main__':
    from PIL import Image
    from ctpn.lib.fast_rcnn.config import cfg_from_file
    cfg_from_file('./ctpn/ctpn/text.yml')
    im = Image.open('./test_images/1.jpg')
    img = np.array(im.convert('RGB'))
    text_recs, img_drawed, img = text_detect(img)
    Image.fromarray(img_drawed).save('result.jpg')
