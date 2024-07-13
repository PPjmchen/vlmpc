from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import copy

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import imageio

torch.set_num_threads(1)



def centers_by_track_batch(tracker, prediction, bbx):
    
    bbx = np.array(bbx).astype(np.int32)
    bbx = (bbx[0][0], bbx[0][1], bbx[1][0] - bbx[0][0], bbx[1][1] - bbx[0][1])
    
    first_frame = True
    
    # read prediction video
    videos = (prediction).astype('uint8')
    num_videos = prediction.shape[0]
    
    seq_len = videos.shape[1]

    for frame_id in range(seq_len):
        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                init_rect = np.tile(bbx,(num_videos,1))
            except:
                exit()
            frame = videos[:, frame_id]

            tracker.init(frame, init_rect)
            first_frame = False
            save_results = [[] for _ in range(videos.shape[0])]
            centers = [[] for _ in range(videos.shape[0])]
        else:
            frame = videos[:, frame_id]
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                
                for v_id in range(frame.shape[0]):
                    sub_video = copy.deepcopy(frame[v_id])
                    bbox = list(map(int, outputs['bbox'][v_id]))
                    cv2.rectangle(sub_video, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 255, 0), 3)
                    save_results[v_id].append(sub_video)
                    # centers[v_id].append([[bbox[0],bbox[1]],[bbox[2]-bbox[0],bbox[3]-bbox[1]]])
                    centers[v_id].append([[bbox[0],bbox[1]],[bbox[2],bbox[3]]])

    # imageio.mimsave('./imgs/track.gif', np.array(save_results[3]), 'GIF', duration = 0.5)
    bbxes = np.array(centers)

    # revise
    bbxes = np.mean(bbxes,1)
    centers_new = bbxes[:,0]
    centers_new[:,0] += (bbxes[:,1,0] / 2)
    centers_new[:,1] += (bbxes[:,1,1] / 2)

    return bbxes, centers_new



if __name__ == '__main__':
    centers_by_track()