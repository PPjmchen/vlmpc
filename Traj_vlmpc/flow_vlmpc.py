import re
import numpy as np
import torch
import os
import cv2
import json
from PIL import Image

# gpt-4v api
from openai import OpenAI
import base64
import httpx
from tqdm import tqdm

from mixed_gaussian import generate_3D_trajectory
from cost_function import value_map
import open3d as o3d

from image2robot import get_world_point_cloud

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt
import supervision as sv

import time

from visualization import visualization_plotly


class VLMPC_flow():
    def __init__(self,args):
        self.num_flows = 50
        self.num_mask_points = 2
        self.trajectory_length = 50
        self.current_center = np.array([127, 341])
        self.sub_goal_center = np.array([422, 336])
        self.avoid = np.array([[252, 305],[249,356],[349,248],[359,352]])
        self.num_kernel = 2

        # TODO: set your camera intrinsic matrix
        self.K =  np.array([[],[],[]])

        # TODO: set your camera extrinsic matrix
        self.trans_matrix = np.array([[],[],[]])
        self.current_obs_path = os.path.join(args.log_dir,"obs.png")
        self.current_depth_path = os.path.join(args.log_dir,"obs.npy")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # sam2_checkpoint = "Segment_anything2/checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        # self.predictor = SAM2ImagePredictor(sam2_model)

        token = "d30f9d063720f581389b97af3d4ff0a8"
        config = Config(token)
        self.client = Client(config)



    def cost_fn(self, world_trajectories, cur_pos_3D, sub_goal_center, depth_map, num_exe, show_map=False):
        """
        world_trajectories.shape: (N,t,3)
        """
        
        
        
        # world_target = image2robot_mech([sub_goal_center[0], sub_goal_center[1]], depth_map,)
        world_target = sub_goal_center # 3D pos
        world_avoids = []
        for i in range(len(self.avoid)):
            # world_avoid = image2robot_mech([self.avoid[i][0], self.avoid[i][1]], depth_map)
            world_avoid = self.avoid[i]
            world_avoids.append(world_avoid)
            
            # world_avoid, _ = self.find_3D_center(depth_map)    # 使用质心
        world_avoids = np.array(world_avoids)

        values, voxel_grid, X, Y, Z = value_map(self.pcd_scene, current_pos=cur_pos_3D, target=world_target, avoid=world_avoids, show_map=show_map, world_trajectories=world_trajectories)

        # Calculate the index of the trajectory in the map
        min_bound = voxel_grid.get_min_bound()
        voxel_size = voxel_grid.voxel_size
        trajectories_voxel_indices = ((world_trajectories - min_bound) // voxel_size).astype(int)

        # Index of recorded penetration
        indexes_filter = []
        for i in range(len(trajectories_voxel_indices)):
            if trajectories_voxel_indices[i].min() < 3:
                indexes_filter.append(i)
        
        trajectories_voxel_indices[...,0] = np.clip(trajectories_voxel_indices[...,0],0,values.shape[0]-1)
        trajectories_voxel_indices[...,1] = np.clip(trajectories_voxel_indices[...,1],0,values.shape[1]-1)
        trajectories_voxel_indices[...,2] = np.clip(trajectories_voxel_indices[...,2],0,values.shape[2]-1)

        # Calculate the score in the value map
        scores = values[trajectories_voxel_indices[...,0],trajectories_voxel_indices[...,1],trajectories_voxel_indices[...,2]]
        
        if scores.shape[1] > num_exe:
            scores = scores[:,:num_exe].sum(1) * 0.6 + scores[:,num_exe:].sum(1) * 0.4
        else:
            scores = scores.sum(1)

        scores[indexes_filter] = 1e5
        print(f"penetration:{indexes_filter}")
        
        cent_point_z = (world_trajectories[:,-1,-1] + world_trajectories[:,0,-1]) / 2
        length = world_trajectories.shape[1]
        true_cent_point_z = world_trajectories[:,length // 2,-1]
        flag = true_cent_point_z < cent_point_z
        
        scores[np.where(flag)[0]] = 1e5

        best_tra_index = scores.argmin()
        best_traj = world_trajectories[best_tra_index]
        
        # plotly 
        fig1, fig2 = visualization_plotly(X, Y, Z, values, self.pcd_scene, world_trajectories, best_traj)
        np.save('world_trajectories.npy',world_trajectories)
        np.save('values.npy',values)
        o3d.io.write_point_cloud('pcd_scene.ply',self.pcd_scene)
        np.save('best_traj.npy',best_traj)
        np.save('X.npy',X)
        np.save('Y.npy',Y)
        np.save('Z.npy',Z)

        return scores, fig1, fig2


    def camera_points_2_world(self, camera_points):
        R = self.trans_matrix[:,:-1]
        T = self.trans_matrix[:,-1]
        world_points = np.dot(camera_points, R.T) + T
        # world_points = R @ camera_points + T

        return world_points


    def generate_3D_tra(self, cur_pos_3D, end_3D, rgb_image, depth_map, covariances, num_traj=30, num_waypoints=120, flat=False):

        start_3D = cur_pos_3D
         
        # end_3D[2] += 0.05
        print(f"start_3D: {start_3D}")

        print(f"end_3D: {end_3D}")
        
        self.pcd_scene = get_world_point_cloud(rgb_image, depth_map) # 经过切割后的pointcloud
        
        trajectories, pcd_trajs = generate_3D_trajectory(start_3D, end_3D, num_traj=num_traj, num_control_points=1, num_curve_points=num_waypoints, pcd_scene=self.pcd_scene, covariances=covariances, flat=flat)

        return trajectories, pcd_trajs


    def find_sub_goal(self, sub_goal_name, log_dir):
        time1 = time.time()
        image_url = self.client.upload_file(self.current_obs_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=sub_goal_name)],
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox, DetectionTarget.Mask]
        )
        self.client.run_task(task)
        predictions = task.result.objects
        obj = predictions[0]
        mask = DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size)
        box = obj.bbox
        cls_name = obj.category.lower().strip()
        
        coords = np.argwhere(mask==1)
        if coords.shape[0] == 0:
            print("No object found in the mask.")
        else:
            x = round(np.mean(coords[:, 1]))
            y = round(np.mean(coords[:, 0]))
        time2 = time.time()
        print(f"dinox time:{time2-time1}")

        # vis
        detections = sv.Detections(
            xyxy = np.array([box]),
            mask = np.array([mask]).astype(bool),
            class_id = np.array([0]),
        )
        img = cv2.imread(self.current_obs_path)
        
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(log_dir, f"{sub_goal_name}_annotated_image_with_mask.jpg"), annotated_frame)
        
        return np.array([x,y])

    
    def find_sub_goal_qwen(self, sub_goal, log_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        import torch
        torch.manual_seed(1234)

        # Note: The default behavior now has injection attack prevention off.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        # 1st dialogue turn
        query = tokenizer.from_list_format([
            {'image': self.current_obs_path}, # Either a local path or an url
            {'text': f'给我{sub_goal}的bounding box'},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        box_content = re.search(r'<box>(.*?)</box>', response)
        
        
        if box_content:
            coordinates = re.findall(r'\((\d+),(\d+)\)', box_content.group(1))
            result = [[int(x), int(y)] for x, y in coordinates]
        else:
            result = []
        result = np.array(result)
        
        image = tokenizer.draw_bbox_on_latest_picture(response, history)
        if image:
            image.save(os.path.join(log_dir, f"{sub_goal}_qwen_image.jpg"))
        else:
            print("no box")
        
        x = round(np.mean(result[:, 0]) / 1024 * 1920)
        y = round(np.mean(result[:, 1]) / 1024 * 1200)
        
        # result[:,0] = result / 1024 * w
        # result[:,1] = result / 1024 * h
        # return result

        return np.array([x,y])
    
    def find_sub_goals(self, sub_goal_name, log_dir, num_goals):
            time1 = time.time()
            image_url = self.client.upload_file(self.current_obs_path)
            task = DinoxTask(
                image_url=image_url,
                prompts=[TextPrompt(text=sub_goal_name)],
                bbox_threshold=0.25,
                targets=[DetectionTarget.BBox, DetectionTarget.Mask]
            )
            self.client.run_task(task)
            predictions = task.result.objects
            centers = []
            boxes = []
            masks = []
            for i in range(num_goals):
                obj = predictions[i]
                mask = DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size)
                box = obj.bbox
                cls_name = obj.category.lower().strip()
                
                coords = np.argwhere(mask==1)
                if coords.shape[0] == 0:
                    print("No object found in the mask.")
                else:
                    x = round(np.mean(coords[:, 1]))
                    y = round(np.mean(coords[:, 0]))
                centers.append(np.array([x,y]))
                boxes.append(box)
                masks.append(mask)
            
            time2 = time.time()
            print(f"dinox time:{time2-time1}")
            # vis
            detections = sv.Detections(
                xyxy = np.array(boxes),
                mask = np.array(masks).astype(bool),
                class_id = np.zeros(num_goals).astype(int),
            )
            img = cv2.imread(self.current_obs_path)
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(log_dir, f"{sub_goal_name}_annotated_image_with_mask.jpg"), annotated_frame)
            
            return centers

    


if __name__ == "__main__":

    pass
