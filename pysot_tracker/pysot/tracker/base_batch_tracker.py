# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg


class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseBatchTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz    
        im_sz = im.shape    # (B, 256, 256, 3)
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = np.maximum(0., -context_xmin).astype(int)
        top_pad = np.maximum(0., -context_ymin).astype(int)
        right_pad = np.maximum(0., context_xmax - im_sz[1] + 1).astype(int)
        bottom_pad = np.maximum(0., context_ymax - im_sz[2] + 1).astype(int)

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        batch_size, r, c, k = im.shape
       
        im_patch = []
        for batch_id in range(batch_size):
            top_pad_sub = top_pad[batch_id]
            bottom_pad_sub = bottom_pad[batch_id]
            left_pad_sub = left_pad[batch_id]
            right_pad_sub = right_pad[batch_id]
            if any([top_pad_sub, bottom_pad_sub, left_pad_sub, right_pad_sub]):
                size = (r + top_pad_sub + bottom_pad_sub, c + left_pad_sub + right_pad_sub, k)
                te_im_sub = np.zeros(size, np.uint8)
                te_im_sub[top_pad_sub:top_pad_sub + r, left_pad_sub:left_pad_sub + c, :] = im[batch_id]
                if top_pad_sub:
                    te_im_sub[0:top_pad_sub, left_pad_sub:left_pad_sub + c, :] = avg_chans[batch_id]
                if bottom_pad_sub:
                    te_im_sub[r + top_pad_sub:, left_pad_sub:left_pad_sub + c, :] = avg_chans[batch_id]
                if left_pad_sub:
                    te_im_sub[:, 0:left_pad_sub, :] = avg_chans[batch_id]
                if right_pad_sub:
                    te_im_sub[:, c + left_pad_sub:, :] = avg_chans[batch_id]
                im_patch_sub = te_im_sub[int(context_ymin[batch_id]):int(context_ymax[batch_id] + 1),
                                int(context_xmin[batch_id]):int(context_xmax[batch_id] + 1), :]
            else:
                im_patch_sub = im[batch_id][int(context_ymin[batch_id]):int(context_ymax[batch_id] + 1),
                            int(context_xmin[batch_id]):int(context_xmax[batch_id] + 1), :]
            # im_patch.append(im_patch_sub)

            # if not np.array_equal(model_sz, original_sz[batch_id]):
            im_patch_sub = cv2.resize(im_patch_sub, (model_sz, model_sz))
            im_patch_sub = im_patch_sub.transpose(2, 0, 1)
            im_patch_sub = im_patch_sub[np.newaxis, :, :, :]
            im_patch_sub = im_patch_sub.astype(np.float32)
            im_patch_sub = torch.from_numpy(im_patch_sub)
            if cfg.CUDA:
                im_patch_sub = im_patch_sub.cuda()
            im_patch.append(im_patch_sub)
        try:
            im_patch = torch.cat(im_patch, 0)
        except:
            import ipdb; ipdb.set_trace()
        
        return im_patch
