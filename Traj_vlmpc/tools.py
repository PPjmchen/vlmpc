import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def bbox_convert_vert_to_xywh(bbox):
    # [[x0,y0], [x1,x1]] -> [x0,y0,w,h]
    return (bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        # torch.backends.cudnn.deterministic = True  #needed
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def dict_to_numpy(d):
    return {
        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in d.items()
    }

def tf_collate_fn(batch):
        x, y = zip(*batch)
        x = torch.stack(x).permute(0, 3, 1, 2).type(torch.FloatTensor)
        y = torch.stack(y)
        return x, y

def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

def draw_bbx(bbx, image_path, text):

    bbx = (np.array(bbx)).astype(int).tolist()
    # b_box upper left
    ptLeftTop = np.array(bbx[0])
    # b_box lower right
    ptRightBottom =np.array(bbx[1])
    # bbox color
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 4

    src = cv2.imread(image_path)
    
    src = np.array(src)
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)


    t_size = cv2.getTextSize(text, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]

    textlbottom = ptLeftTop + np.array(list(t_size))

    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)

    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)

    cv2.putText(src, text , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    cv2.imwrite(image_path[:-4]+'_bbox.png', src)

import numpy as np
import open3d as o3d
from PIL import Image

def get_camera_intrinsics():
    focal_length = 732.9993  
    width = 640          
    height = 480         
    cx = width / 2.0     
    cy = height / 2.0   

    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]])
    
    return K

def get_point_cloud_coordinate(u, v, depth_map, K):
    depth = depth_map[v, u]
    
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    
    x = x * depth
    y = y * depth
    z = depth
    
    return np.array([x, y, z])

def get_camera_coordinate(u ,v, depth_map, K):
    pixel = np.array([u, v, 1])
    Z = depth_map[v, u]

    camera_coord = Z * np.linalg.inv(K) @ pixel

    return camera_coord


def get_camera_coordinates(pixel_coords, depth_map, K):

    ones = np.ones((pixel_coords.shape[0],1))
    pixel_coords = np.hstack((pixel_coords,ones))


    depth = depth_map[pixel_coords[:,1].astype(int),pixel_coords[:,0].astype(int)]


    camera_coords = depth[:,np.newaxis] * (np.linalg.inv(K) @ (pixel_coords).T).T

    return camera_coords




def calculate_transformation_matrix_3d(world_points, camera_points):

    world_points = np.array(world_points)
    camera_points = np.array(camera_points)


    centroid_world = np.mean(world_points, axis=0)
    centroid_camera = np.mean(camera_points, axis=0)


    world_points_centered = world_points - centroid_world
    camera_points_centered = camera_points - centroid_camera


    H = camera_points_centered.T @ world_points_centered


    U, S, Vt = np.linalg.svd(H)
    

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T


    t = centroid_world - R @ centroid_camera


    transformation_matrix = np.hstack((R, t.reshape(3, 1)))

    return transformation_matrix


def get_transformation_matrix():

    camera_points = [
                    [1, 0.0, 0.0],
                    [0.0, 2, 0.0],
                    [0.0, 0.0, 2]]

    world_points = [
                    [2, 1, 2],
                    [3, 0, 0.2679491],
                    [0.2679491, 0, 1]]



    transformation_matrix = calculate_transformation_matrix_3d(world_points, camera_points)
    return transformation_matrix

def caculate_world_point(u, v, depth_map):
    K = get_camera_intrinsics()
    camera_point_coord = get_camera_coordinate(u ,v, depth_map, K)

    trans_matrix = get_transformation_matrix()
    R = trans_matrix[:,:-1]
    T = trans_matrix[:,-1]
    world_point = R @ camera_point_coord + T

    return world_point