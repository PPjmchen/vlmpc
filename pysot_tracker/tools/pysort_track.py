from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

def centers_by_track(prediction, bbx):
    checkpoint_path = '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/model.pth'
    config_path =  '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/config.yaml'
    bbx = (bbx / 4).astype(np.uint8)
    # load config
    cfg.merge_from_file(config_path)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(checkpoint_path,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    sample_centers = []
    for frames in prediction["rgb"]:
        first_frame = True
        bbxes = []
        for frame in frames:
            if first_frame:
                try:
                    init_rect = (bbx[0][0], bbx[0][1], bbx[1][0] - bbx[0][0], bbx[1][1] - bbx[0][1])   # (x,y,h,w)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
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
                    bbox = list(map(int, outputs['bbox']))
                    bbox = [[bbox[0],bbox[1]],[bbox[0]+bbox[2],bbox[1]+bbox[3]]]
                    bbxes.append(bbox)
        centers = np.mean(np.array(bbxes),1)
        center = np.mean(centers,0)
        # import ipdb;ipdb.set_trace()
        sample_centers.append(center)
    return np.array(sample_centers)

def centers_by_track_batch(prediction, bbx):
    checkpoint_path = '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/model.pth'
    config_path =  '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/config.yaml'
    bbx = (bbx / 4).astype(np.uint8)
    bbx = (bbx[0][0], bbx[0][1], bbx[1][0] - bbx[0][0], bbx[1][1] - bbx[0][1])
    # load config
    import ipdb;ipdb.set_trace()
    cfg.merge_from_file(config_path)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(checkpoint_path,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    
    first_frame = True
    
    # read prediction video
    videos = prediction["rgb"]
    num_videos = prediction["rgb"].shape[0]
    
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
                    centers[v_id].append([[bbox[0],bbox[1]],[bbox[2]-bbox[0],bbox[3]-bbox[1]]])
    centers = np.array(centers)
    centers = np.mean(np.mean(centers,1),1)
    import ipdb;ipdb.set_trace()
    return centers



if __name__ == '__main__':
    centers_by_track()