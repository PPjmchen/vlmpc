import numpy as np
import torch
import imageio
import os
import cv2
import json

from sampler import CorrelatedNoiseSampler
from prompt_gpt import get_subtasks, get_interactive_object
import matplotlib.pyplot as plt

# gpt-4v api
from openai import OpenAI
import base64
import httpx

from tools import bbox_convert_vert_to_xywh

from pysot_tracker.tools.tracker_bbx import bbx_tracker
from pysot_tracker.tools.pysort_track_batch import centers_by_track_batch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from detect_bbx import det_bbox

import time

class VLMPC():
    def __init__(self, video_prediction_model, action_dim, action_horizon, det_model, init_action=None, init_std=0.01, num_samples=200,
                 logdir=None, logger=None, task=None, prompt_json='./prompt.json', plan_freq=3, zoom = 0.02, history_rate=0.5, ratio_tar_obj=0.5):
        self.video_prediciton_model = video_prediction_model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.std = np.ones([self.action_horizon, self.action_dim]) * init_std
        self.sampler = CorrelatedNoiseSampler(a_dim=action_dim, beta=0.5, horizon=action_horizon)
        self.num_samples = num_samples
        self.log_dir= logdir
        self.logger = logger
        self.task = task
        self.subtasks = []
        self.plan_freq = plan_freq
        self.obs_list = []
        self.zoom = zoom
        self.history_rate = history_rate
        self.ratio_tar_obj = ratio_tar_obj
        self.det_model = det_model
        self.det_model.conf = 0.5
        self.classes = self.det_model.names
        self.classes_rev = {v: k for k, v in self.classes.items()}
 
        self.gpt_client = OpenAI(
                    # base_url="https://hk.xty.app/v1",
                    base_url="http://47.76.75.25:9000/v1",
                    api_key="sk-itQ7BCQZjpcvqKfd035b9c9f475d4b0aA6452472Cf28DfF0",
                    http_client=httpx.Client(
                    base_url="http://47.76.75.25:9000/v1",
                    follow_redirects=True,
                    ),
                )

        self.end_tracker = self.get_tracker()
        self.obj_tracker = self.get_tracker()
        self.video_prediction_tracker = self.get_tracker(mode='batch')

        if init_action != None:
            self.init_action = init_action
        else:
            self.init_action = np.zeros(self.action_dim)

        self.num_steps = 0

        if self.task == 'push_corner':
            self.corner_pos = np.array([310,164])
        
        self.best_actions = None

    def get_tracker(self, mode='single'):
        cfg.merge_from_file('./pysot_tracker/experiments/siamrpn_alex_dwxcorr/config.yaml')
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda:0' if cfg.CUDA else 'cpu')

        if mode == 'batch':
            cfg.TRACK.TYPE = 'SiamRPNBatchTracker'
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load('./pysot_tracker/experiments/siamrpn_alex_dwxcorr/model.pth', map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        tracker = build_tracker(model)
        return tracker

    def reset(self, init_frame=None):
        time0 = time.time()
        
        self.init_frame = init_frame
        if self.num_steps == 0:
            response_subtasks = get_subtasks(image_path='%s/init_task.png' % self.log_dir, client=self.gpt_client, task=self.task)
        else:
            response_subtasks = get_subtasks(image_path='%s/current_obs.png' % self.log_dir, client=self.gpt_client, task=self.task, excluding=self.current_interactive_object)
        time1 = time.time()

        self.logger.info(response_subtasks)

        self.subtasks.append(response_subtasks)
        
        self.history_mu = np.zeros([self.action_horizon, self.action_dim])
        
        self.current_subtask = response_subtasks
        self.current_interactive_object = get_interactive_object(self.gpt_client, self.current_subtask)
        
        self.logger.info('current interactive object is: %s' % str(self.current_interactive_object))
        time2 = time.time()
        self.logger.info("get subtask time: %s" % (time1 - time0))
        self.logger.info("get intereactive object time: %s" % (time2 - time1))
        return np.zeros(self.action_dim)

    def cost_fn(self, frames, predictions):
        time0 = time.time()
        current_frame = frames[-1]
        arm_bbxes, arm_centers = centers_by_track_batch(self.video_prediction_tracker, predictions * 255, self.current_end_bbox)
        subtask_obj_bbxes, subtask_obj_centers = centers_by_track_batch(self.video_prediction_tracker, predictions * 255, self.subtask_obj_bbox)
        time1 = time.time()
        
        # define a target bbx 
        if self.task == 'push_corner':
            target_center = self.corner_pos

            # if reach subtask_obj, threshold = 0.5
            iou = calculate_iou(arm_bbxes,subtask_obj_bbxes)
            threshold = 1e-8
            flags = iou > threshold
            
            # TODO: calculate the cost of videos
            dis_arm_obj = compute_dis(arm_centers,subtask_obj_centers)
            dis_obj_target = compute_dis(subtask_obj_centers,target_center)

            origin_center = (np.array(self.subtask_obj_bbox[0])+np.array(self.subtask_obj_bbox[1])) / 2
            origin_dis = np.linalg.norm(origin_center-target_center)

            # TODO: avoid the obstacles
            obstacle_dises = []
            for i in range(8):
                if i == self.classes_rev[self.current_interactive_object]:
                    continue
                obstacle_bbx = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str=self.classes[i])
                if obstacle_bbx == 0:
                    continue
                obstacle_center = np.array([(obstacle_bbx[0][0] + obstacle_bbx[1][0]) / 2, (obstacle_bbx[0][1] + obstacle_bbx[1][1]) / 2])
                dis_arm_obstacle = compute_dis(arm_centers, obstacle_center)
                obstacle_dises.append(dis_arm_obstacle)

            dis_obstacles_average = np.sum(np.array(obstacle_dises),0) / len(obstacle_dises)

            
            if True in (dis_obj_target < origin_dis):
                cost = dis_obj_target - dis_obstacles_average
            else:
                cost = (dis_arm_obj - dis_obstacles_average) * [not x for x in flags] + (dis_obj_target * self.ratio_tar_obj + dis_arm_obj * self.ratio_tar_obj - dis_obstacles_average * 0.25) * flags
        time2 = time.time()
        self.logger.info("time track_center: %s " % (time1 - time0))
        self.logger.info("time cacu: %s " % (time2 - time1))
        return cost.argsort()

    def plan(self, frames):
        
        action_samples = self.sampler.sample_actions(self.num_samples, self.mu, self.std)
        action_samples = np.clip(action_samples, -0.02, 0.02)
        batch = {
            'video': frames[None].repeat(self.num_samples, 1, 1, 1, 1).numpy(),
            'actions': action_samples
        }
        
        if self.num_steps % self.plan_freq == 0:
            predictions = self.video_prediciton_model(batch)['rgb']
            
            scores = self.cost_fn(frames, predictions) # better -> worse
            
            best_action = action_samples[scores][0][:self.plan_freq]                   # select the first 5 actions
            self.history_action_samples = action_samples[0][self.plan_freq:]

            self.history_mu = np.zeros([self.action_horizon, self.action_dim])
            self.history_mu[:self.action_horizon-self.plan_freq] = self.history_action_samples
            self.history_mu[-self.plan_freq:] = self.history_action_samples[-1]
            
        return best_action
        
    def check_subtask(self):
        subtask_finish_status = False
        if self.task == 'push_corner':
            if np.linalg.norm(self.obj_pos - self.corner_pos) <= 100:

                self.logger.info('subtask object are moved to the target region')
                subtask_finish_status = True
            return subtask_finish_status

    def check_overall_task(self, image_path):
        for i in range(8):

            obj_bbx = det_bbox(image_path=image_path, model=self.det_model, obj_str=self.classes[i])
            if obj_bbx == 0:
                continue
            obj_pos = np.array([(obj_bbx[0][0] + obj_bbx[1][0]) / 2, (obj_bbx[0][1] + obj_bbx[1][1]) / 2])
            if np.linalg.norm(obj_pos - self.corner_pos) <= 100:
            # if self.corner_pos[0] - obj_pos[0] <= 120 and self.corner_pos[1] - obj_pos[1] <= 74:
                self.logger.info("%s has been moved to the target corner." % self.classes[i])
                state = True
            else:
                self.logger.info("%s has not been moved to target corner." % self.classes[i])
                return False        
        return state

    def act(self, num_steps, frames, subtask_finish_status=False):
        self.num_steps = num_steps


        if self.num_steps > 0:
            subtask_finish_status = self.check_subtask()

        # TODO: action sampling with VLM
        self.current_obs_image_path = os.path.join(self.log_dir, 'current_obs.png')
        plt.imsave(self.current_obs_image_path, frames[1].numpy())

        # check overall task
        if subtask_finish_status == True:
            self.logger.info("Check overall task.")
            overall_task_status = self.check_overall_task(self.current_obs_image_path)
            if overall_task_status == True:
                self.logger.info("All the blocks have been moved to the corner.")
                imageio.mimsave('%s/current_obs_with_boxes.gif' % self.log_dir, self.obs_list, 'GIF', duration = 1)
                return 0
            self.reset()            

        
        if self.num_steps == 0 or subtask_finish_status == True: # using VLM in the beginning of subtasks
            self.subtask_obj_bbox = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str=self.current_interactive_object)
            while type(self.subtask_obj_bbox) == int:
                self.logger.info("can not find %s, I will reset." % self.current_interactive_object)
                self.reset()
                self.subtask_obj_bbox = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str=self.current_interactive_object)
            self.obj_pos = np.array([(self.subtask_obj_bbox[0][0] + self.subtask_obj_bbox[1][0]) / 2, (self.subtask_obj_bbox[0][1] + self.subtask_obj_bbox[1][1]) / 2])
            self.last_obj_pos = self.obj_pos
            self.last_subtask_obj_bbox = self.subtask_obj_bbox

            self.current_end_bbox = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str="end effector") # [[x0,y0], [x1,y1]]
            self.end_pos = np.array([(self.current_end_bbox[0][0] + self.current_end_bbox[1][0]) / 2, self.current_end_bbox[1][1]])
            
            self.end_tracker.init(cv2.imread(self.current_obs_image_path), bbox_convert_vert_to_xywh(self.current_end_bbox))
            self.obj_tracker.init(cv2.imread(self.current_obs_image_path), bbox_convert_vert_to_xywh(self.subtask_obj_bbox))

            
        else: # using tracker

            # find end
            self.current_end_bbox = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str="end effector") # [[x0,y0], [x1,y1]]
            if self.current_end_bbox == 0:
                tracker_outputs_end = self.end_tracker.track(cv2.imread(self.current_obs_image_path))['bbox'] # [x0,y0,w,h]
                self.current_end_bbox = np.array([[tracker_outputs_end[0], tracker_outputs_end[1]],[tracker_outputs_end[0]+tracker_outputs_end[2], tracker_outputs_end[1]+tracker_outputs_end[3]]]).astype(np.int64)
                # self.end_tracker.init(cv2.imread(self.current_obs_image_path), tuple(tracker_outputs_end))
            self.end_tracker.init(cv2.imread(self.current_obs_image_path), bbox_convert_vert_to_xywh(self.current_end_bbox))
            self.end_pos = np.array([(self.current_end_bbox[0][0] + self.current_end_bbox[1][0]) / 2, self.current_end_bbox[1][1]]) 
                

            # find obj
            self.subtask_obj_bbox = det_bbox(image_path=self.current_obs_image_path, model=self.det_model, obj_str=self.current_interactive_object)
            if self.subtask_obj_bbox == 0:
                self.logger.info("Can't detect the obj. use tracking.")
                tracker_outputs_obj = self.obj_tracker.track(cv2.imread(self.current_obs_image_path))['bbox']
                self.subtask_obj_bbox = np.array([[tracker_outputs_obj[0], tracker_outputs_obj[1]],\
                                     [tracker_outputs_obj[0]+tracker_outputs_obj[2], tracker_outputs_obj[1]+tracker_outputs_obj[3]]]).astype(np.int64)
            self.obj_pos = np.array([(self.subtask_obj_bbox[0][0] + self.subtask_obj_bbox[1][0]) / 2, (self.subtask_obj_bbox[0][1] + self.subtask_obj_bbox[1][1]) / 2])
            
            # judge if jump
            if np.linalg.norm(self.obj_pos - self.last_obj_pos) > 80:
                self.logger.info("First jump. Use tracking.")
                tracker_outputs_obj = self.obj_tracker.track(cv2.imread(self.current_obs_image_path))['bbox']
                self.subtask_obj_bbox = np.array([[tracker_outputs_obj[0], tracker_outputs_obj[1]], [tracker_outputs_obj[0]+tracker_outputs_obj[2], tracker_outputs_obj[1]+tracker_outputs_obj[3]]]).astype(np.int64)
                self.obj_pos = np.array([(self.subtask_obj_bbox[0][0] + self.subtask_obj_bbox[1][0]) / 2, (self.subtask_obj_bbox[0][1] + self.subtask_obj_bbox[1][1]) / 2])
            
            # judge if jump twice
            if np.linalg.norm(self.obj_pos - self.last_obj_pos) > 80:
                self.logger.info("Second jump. Use the last pos")
                self.subtask_obj_bbox = self.last_subtask_obj_bbox
                self.obj_pos = self.last_obj_pos

            self.obj_tracker.init(cv2.imread(self.current_obs_image_path), bbox_convert_vert_to_xywh(self.subtask_obj_bbox))
            self.last_obj_pos = self.obj_pos
            self.last_subtask_obj_bbox = self.subtask_obj_bbox
            
        
        current_obs = cv2.imread(self.current_obs_image_path)
        cv2.rectangle(current_obs, tuple(self.subtask_obj_bbox[0]), tuple(self.subtask_obj_bbox[1]),  (0, 255, 0), 2, 4)
        cv2.rectangle(current_obs, tuple(self.current_end_bbox[0]), tuple(self.current_end_bbox[1]),  (0, 0, 255), 2, 4)
        cv2.circle(current_obs, np.array(self.end_pos).astype(int), 3, (0, 0, 255), 0)
        cv2.circle(current_obs, np.array(self.obj_pos).astype(int), 3, (0, 255, 0), 0)
        text = str(self.current_interactive_object)
        cv2.putText(current_obs, text, (15,15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        current_obs = cv2.cvtColor(current_obs, cv2.COLOR_BGR2RGB)

        cv2.imwrite("%s/frame_bbx_%s.png" %(self.log_dir , self.num_steps), current_obs[...,::-1])
        self.obs_list.append(current_obs)
        # imageio.mimsave('%s/current_obs_with_boxes.gif' % self.log_dir, self.obs_list, 'GIF', duration = 1)
        
        if self.best_actions is None or len(self.best_actions) == 0:
            
            # NOTE: Phi_S can be simply replaced by this for acceleration
            moving_direction = (self.obj_pos - self.end_pos)
            moving_direction /= np.sqrt(moving_direction[0]**2 + moving_direction[1]**2)
            moving_direction *= self.zoom

            mu = moving_direction
            self.mu = np.zeros([self.action_horizon, self.action_dim])
            self.mu[:, ] = mu
            self.mu = self.history_rate * self.mu + (1-self.history_rate) * self.history_mu
            
            best_action = self.plan(frames)
            best_action = convert_env_action(best_action)

            self.best_actions = best_action
        
        exec_action = self.best_actions[0]
        self.best_actions = np.delete(self.best_actions, 0, axis=0)
        
        return exec_action


def convert_env_action(action):
    new_action = np.zeros_like(action)
    new_action[:,0] = action[:,1]
    new_action[:,1] = action[:,0]
    
    return new_action

def calculate_iou(boxes1, boxes2):
    
    

    x1 = boxes1[:, 0, 0]
    y1 = boxes1[:, 0, 1]
    w1 = boxes1[:, 1, 0]
    h1 = boxes1[:, 1, 1]
    x1_br = x1 + w1
    y1_br = y1 + h1
    
    x2 = boxes2[:, 0, 0]
    y2 = boxes2[:, 0, 1]
    w2 = boxes2[:, 1, 0]
    h2 = boxes2[:, 1, 1]
    x2_br = x2 + w2
    y2_br = y2 + h2
    
    

    x_left = np.maximum(x1, x2)
    y_top = np.maximum(y1, y2)
    x_right = np.minimum(x1_br, x2_br)
    y_bottom = np.minimum(y1_br, y2_br)
    
    intersection_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
    
    

    box1_area = (x1_br - x1) * (y1_br - y1)
    box2_area = (x2_br - x2) * (y2_br - y2)
    union_area = box1_area + box2_area - intersection_area
    
    

    iou = intersection_area / union_area
    return iou

def compute_dis(centers1,centers2):
    return np.linalg.norm(centers1 - centers2, axis=1)