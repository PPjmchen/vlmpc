# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_batch_tracker import SiameseBatchTracker


class SiamRPNBatchTracker(SiameseBatchTracker):
    def __init__(self, model):
        super(SiamRPNBatchTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta[:,None,:,:,:].permute(0, 2, 3, 4, 1).contiguous().view(delta.shape[0], 4, -1)
        delta = delta.data.cpu().numpy()
        delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        delta[:, 2, :] = np.exp(delta[:, 2, :]) * anchor[:, 2]
        delta[:, 3, :] = np.exp(delta[:, 3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score[:,None,:,:,:].permute(0, 2, 3, 4, 1).contiguous().view(score.shape[0], 2, -1).permute(0, 2, 1)
        score = F.softmax(score, dim=2).data[:, :, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.center_pos = np.array([bbox[:,0]+(bbox[:,2]-1)/2,
                                    bbox[:,1]+(bbox[:,3]-1)/2])
        self.size = np.array([bbox[:,2], bbox[:,3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 0)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 0)
        s_z = np.round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(1, 2))
       
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 0)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, 0)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    np.round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[:, 2, :], pred_bbox[:, 3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z))[:,np.newaxis])
        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1])[:,np.newaxis] /
                     (pred_bbox[:, 2, :]/pred_bbox[:, 3, :]))
        
        
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore, 1)

        bbox = pred_bbox[np.arange(pred_bbox.shape[0]), :, best_idx] / scale_z[:,np.newaxis]
        lr = penalty[np.arange(penalty.shape[0]), best_idx] * score[np.arange(score.shape[0]), best_idx] * cfg.TRACK.LR

        cx = bbox[:, 0] + self.center_pos[0]
        cy = bbox[:, 1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[:, 2] * lr
        height = self.size[1] * (1 - lr) + bbox[:, 3] * lr

        # clip boundary
        for j in range(cx.shape[0]): 
            cx[j], cy[j], width[j], height[j] = self._bbox_clip(cx[j], cy[j], width[j],
                                                    height[j], img.shape[1:3])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        
        bbox = np.array([cx - width / 2,
                cy - height / 2,
                width,
                height]).transpose(1,0)
        best_score = score[np.arange(score.shape[0]), best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
