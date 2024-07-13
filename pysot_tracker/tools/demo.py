from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import imageio
import numpy as np
from glob import glob



import copy

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames_new(video_name):
    all_frames = []
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4') or video_name.endswith('gif'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                all_frames.append(frame)
            else:
                return(all_frames)
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def get_video(video_name):
    all_frames = []
    cap = cv2.VideoCapture(video_name)
    while True:
        ret, frame = cap.read()
        if ret:
            all_frames.append(frame)
        else:
            return(all_frames)

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    save_results = []

    video_1 = np.array(get_video('demo/org1.gif'))[np.newaxis]
    video_2 = np.array(get_video('demo/org199.gif'))[np.newaxis]
    # import ipdb;ipdb.set_trace()
    
    # video_3 = copy.deepcopy(video_1)

    videos = np.concatenate([video_1, video_2, ], 0)
    # videos = video_1

    seq_len = videos.shape[1]

    for frame_id in range(seq_len):
        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                init_rect = np.array([[142, 48, 17, 57],
                                      [142, 48, 17, 57],
                                      ])
            except:
                exit()
            frame = videos[:, frame_id]
            tracker.init(frame, init_rect)
            first_frame = False
            save_results = [[] for _ in range(videos.shape[0])]
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
                    print('video_%s' % v_id)
                    print(bbox)
    import ipdb; ipdb.set_trace()
    for v_id in range(frame.shape[0]):
        imageio.mimsave('./demo/output_org_%s.gif' % v_id, save_results[v_id], 'GIF', duration = 0.5)
        # imageio.mimsave('/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/temp199.gif', np.array(save_results)[199], 'GIF', duration = 0.5)

if __name__ == '__main__':
    main()
