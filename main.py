import os
import sys
import torch
import numpy as np
import argparse
import imageio
import cv2

import tensorflow_datasets as tfds

from tools import seed_all, decode_inst
from logger import setup_logger
from datetime import datetime

from absl import app
import matplotlib.pyplot as plt

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block, block1_to_corner

from video_interface import DMVFNActModel
from vlmpc import VLMPC
from multiprocessing import Pool
import time


sys.path.append('./models/dmvfn/')
sys.path.append('./pysot_tracker')

def mpc_loop(vlmpc, init_frame, frame, env):

    last_frame = init_frame
    current_frame = frame
    loop_obs = [init_frame, frame]

    frames = torch.cat([torch.from_numpy(last_frame[None]), torch.from_numpy(current_frame[None])])

    num_steps = 0
    while num_steps < args.max_traj_length:
        logger.info('step %s' %num_steps)
        time0 = time.time()
        best_action = vlmpc.act(num_steps, frames)

        # judge if task success
        if type(best_action) == int:
            logger.info("All the blocks have been pushed to the corner.")
            break

        time1 = time.time()
        logger.info('exec action is: %s' % (best_action) )
        env.step(best_action)
        time2 = time.time()

        last_frame = current_frame
        current_frame = env._render_camera(image_size=env._image_size)
        frames = torch.cat([torch.from_numpy(last_frame[None]), torch.from_numpy(current_frame[None])])

        loop_obs.append(current_frame)
        cv2.imwrite("%s/frame_%s.png" %(vlmpc.log_dir , vlmpc.num_steps), current_frame[...,::-1])
        # imageio.mimsave('%s/output.gif' % args.log_dir, loop_obs, 'GIF', duration = 0.5)
        num_steps += 1
        time3 = time.time()
        logger.info("one step: %s" % (time3 - time0))

    imageio.mimsave('%s/output.gif' % args.log_dir, loop_obs, 'GIF', duration = 0.5)
    imageio.mimsave('%s/current_obs_with_boxes.gif' % vlmpc.log_dir, vlmpc.obs_list, 'GIF', duration = 0.1)

def read_dataset(path, skip=0):
        builder = tfds.builder_from_directory(path)

        episode_ds = builder.as_dataset(split='train')
        episode = next(iter(episode_ds.skip(0).take(1)))
        frames = []
        actions = []
        for step in episode['steps'].as_numpy_iterator():
            frames.append(torch.from_numpy(cv2.resize(step['observation']['rgb'], (320, 180), interpolation=cv2.INTER_AREA)))
            actions.append(torch.from_numpy(step['action']))
        return frames, actions
        # return None
def main(args):

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block1_to_corner.Block1ToCornerLocationReward,
        control_frequency=10.0,
        seed=args.seed
    )
    init_obs = env.reset()
    instruction = decode_inst(init_obs['instruction'])
    init_frame = init_obs['rgb']
    plt.imsave('%s/init_task.png' % args.log_dir, env._render_camera(image_size=env._image_size))
    
    if args.model == 'dmvfn_action_2dim':
        model = DMVFNActModel(checkpoint_file=args.checkpoint_file, action_dim=args.action_dim, n_past=2, max_batch_size=200)
        model.model.dmvfn.eval()
    
    det_model = torch.hub.load('yolov5', 'custom', path=args.det_path, source='local')

    vlmpc = VLMPC(video_prediction_model=model, action_dim=args.action_dim, det_model=det_model, action_horizon=args.action_horizon, logdir=args.log_dir, logger=logger, task=args.task, plan_freq=args.plan_freq, init_std=args.init_mean, zoom=args.zoom, history_rate=args.history_rate, ratio_tar_obj=args.ratio_tar_obj, num_samples=args.num_samples)
    init_action = vlmpc.reset(init_frame)
    env.step(init_action)
    frame = env._render_camera(image_size=env._image_size)

    mpc_loop(vlmpc, init_frame, frame, env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VLMPC parameters')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, )
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='dmvfn_action_2dim')
    parser.add_argument('--checkpoint_file', type=str, default='ckpts/dmvfn_499.pkl')
    parser.add_argument('--action_dim', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=20)
    parser.add_argument('--max_traj_length', type=int, default=2000)
    parser.add_argument('--task', type=str, choices=['make_line', 'push_corner', 'group_color'])
    parser.add_argument('--plan_freq', type=int, default=5)
    parser.add_argument('--init_mean', type=float, default=0.01)
    parser.add_argument('--zoom', type=float, default=0.02)
    parser.add_argument('--history_rate', type=float, default=0.5)
    parser.add_argument('--ratio_tar_obj', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--det_path', type=str, default='yolov5/runs/train/adamW_yolov5s_2230/weights/best.pt')

    args = parser.parse_args()

    seed_all(args.seed)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = os.path.join(args.log_root, '%s_%s'%(args.tag, stamp))
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(output=args.log_dir, color=True, name='vlmpc')

    main(args)