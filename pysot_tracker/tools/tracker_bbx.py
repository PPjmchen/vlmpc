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



def bbx_tracker(frame, bbx):
    
    checkpoint_path = '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/model.pth'
    config_path =  '/home/cjm/Projects/3DxGrasp/vp2_ur5/vlmpc/vp2_ur5/mpc/pysot/experiments/siamrpn_alex_dwxcorr/config.yaml'
    
    bbx = bbx.astype(np.uint8)
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
    
    # tracker init
    init_rect = np.array([[bbx[0][0], bbx[0][1], bbx[1][0] - bbx[0][0], bbx[1][1] - bbx[0][1]]])   # (x,y,h,w)
    tracker.init(frame, init_rect)
    
    return tracker
    sample_centers = []