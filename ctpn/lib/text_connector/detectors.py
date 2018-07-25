#-*- coding:utf-8 -*-
import numpy as np
from ctpn.lib.fast_rcnn.nms_wrapper import nms
from ctpn.lib.fast_rcnn.config import cfg
from .other import normalize
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented
from .text_connect_cfg import Config as TextLineCfg


class TextDetector:
    def __init__(self):
        self.mode= cfg.TEST.DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector=TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector=TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, size):
        # 删除得分较低的proposal，分数必须高于0.7，得到对应的索引
        keep_inds=np.where(scores>TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        # 获得对应索引的文本框和对应的分数
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序, ravel是行序优先的打平，与flatten不同的是，ravel返回的是对原对象的修改
        # argsort针对数组返回从小到大的索引值，::-1 相当于取反
        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        keep_inds=nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 获取检测结果
        scores=normalize(scores)
        # 在这里进行文本线的绘制
        text_recs=self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        
        # 过滤boxes
        # todo text_recs不知道8个数据都是啥！
        keep_inds=self.filter_boxes(text_recs)
        text_lines=text_recs[keep_inds]
        
        # 对lines做nms
        # TODO npm是啥?
        if text_lines.shape[0] != 0:
            keep_inds=nms(text_lines, TextLineCfg.TEXT_LINE_NMS_THRESH)
            text_lines=text_lines[keep_inds]
        
        return text_lines

    def filter_boxes(self, boxes):
        """
        针对9个数据，前8个是框体相关的，第9个数据是分值

        :param boxes:
        :return:
        """
        heights=np.zeros((len(boxes), 1), np.float)
        widths=np.zeros((len(boxes), 1), np.float)
        scores=np.zeros((len(boxes), 1), np.float)
        index=0
        for box in boxes:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1

        return np.where((widths / heights > TextLineCfg.MIN_RATIO) & (scores > TextLineCfg.LINE_MIN_SCORE) &
                          (widths > (TextLineCfg.TEXT_PROPOSALS_WIDTH * TextLineCfg.MIN_NUM_PROPOSALS)))[0]
