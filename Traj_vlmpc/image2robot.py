import numpy as np
from MechMindCamera import MechMindCam
import matplotlib.pyplot as plt
import open3d as o3d

def image2robot_mech(center, depth):
    x, y= center
    cam_pose = np.loadtxt('mech_param/kinect_camera_pose.txt', delimiter=' ')
    cam_intrinsics = np.loadtxt('mech_param/mech_camera_params.txt', delimiter=' ')
    cam_depth_scale = np.loadtxt('mech_param/kinect_camera_depth_scale.txt', delimiter=' ')

    camera_depth_img = depth
    click_z = camera_depth_img[y][x] * cam_depth_scale
    click_x = np.multiply(x - cam_intrinsics[0][2], click_z / cam_intrinsics[0][0])
    click_y = np.multiply(y - cam_intrinsics[1][2], click_z / cam_intrinsics[1][1])
    if click_z == 0:
        print('bad depth value!!!')
        return (0,0,0)
    click_point = np.asarray([click_x, click_y, click_z])
    click_point.shape = (3, 1)
    # Convert camera to robot coordinates
    camera2robot = cam_pose
    target_position = np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:]
    target_position = target_position[0:3, 0] # (x,y,z) in robot workspace
    return target_position

def downsample_point_cloud(point_cloud, voxel_size=0.001):

    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    return downsampled_cloud


def get_world_point_cloud(rgb, depth, vis=False):
    
    K = np.loadtxt('mech_param/mech_camera_params.txt', delimiter=' ')
    trans_matrix = np.loadtxt("mech_param/kinect_camera_pose.txt")[:3]
    cam_depth_scale = np.loadtxt('mech_param/kinect_camera_depth_scale.txt', delimiter=' ')
    
    
    
    camera_points_3d, colors = get_camera_points(rgb, depth, K, cam_depth_scale)

    R = trans_matrix[:,:-1]
    T = trans_matrix[:,-1]
    world_points = np.dot(camera_points_3d, R.T) + T
    
    mask1 = world_points[:,2] >= 0.25   # clip the table points
    mask2 = world_points[:,2] <= 0.8    # clip the top
    mask = mask1 & mask2
    world_points = world_points[mask]
    colors = colors[mask]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(world_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # downsample
    point_cloud = downsample_point_cloud(point_cloud)
    
    if vis:
        o3d.visualization.draw_geometries([point_cloud])
    return point_cloud
    
def get_camera_points(rgb_image, depth_map, K, cam_depth_scale):
    height, width = depth_map.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    u = u.flatten()
    v = v.flatten()
    depth = depth_map.flatten() * cam_depth_scale
    
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    
    x = x * depth
    y = y * depth
    z = depth

    points_3d = np.stack((x, y, z), axis=-1)
    
    if rgb_image.shape[2] == 4:
        colors = rgb_image[v, u, :3].astype(np.float64) / 255.0
    else:
        colors = rgb_image[v, u, :].astype(np.float64) / 255.0
        
    return points_3d, colors

if __name__ == '__main__':
    pass
    
    
    
    





