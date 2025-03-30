import open3d as o3d
import plotly
import plotly.graph_objects as go
import numpy as np


def generate_point_cloud(rgb_image, depth_map, K):
    height, width = depth_map.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    u = u.flatten()
    v = v.flatten()
    depth = depth_map.flatten()
    
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


def get_world_point_cloud(rgb_image, depth_map, K, trans_matrix):
    camera_points_3d, colors = generate_point_cloud(rgb_image, depth_map, K)

    R = trans_matrix[:,:-1]
    T = trans_matrix[:,-1]
    world_points = np.dot(camera_points_3d, R.T) + T
    mask = world_points[:,2]>= 0.66
    world_points = world_points[mask]
    colors = colors[mask]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(world_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def get_value_data(X,Y,Z,values):

    data_value=[go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=0.3, # needs to be small to see through all surfaces
        surface_count=80, # needs to be a large number for good volsume rendering
        colorscale = "Jet",
        ),]
    
    return data_value


def get_traj_data(trajectories,color,width=1.5):
    # trajtories.shape = (N,t,3)
    trajectories = trajectories.reshape(-1,3)
    data=[go.Scatter3d(x=trajectories[:,0], y=trajectories[:,1], z=trajectories[:,2],
                                   mode='markers',
                                   marker=dict(
                                            size=width,  
                                            color=color,  
                                            opacity=0.8  
                                        ))]
    
    return data


def get_best_traj_data(trajectories, color, width=10):
    # trajectories.shape = (N, t, 3)
    trajectories = trajectories.reshape(-1, 3)
    
    num_points = len(trajectories)
    colorscale = ['red', 'red']
    color_vals = np.linspace(0, 1, num_points)

    data = [go.Scatter3d(
        x=trajectories[:, 0], 
        y=trajectories[:, 1], 
        z=trajectories[:, 2],
        mode='lines',  
        line=dict(
            width=width, 
            color=color,  
        )
    )]
    return data

def get_point_data(point_cloud):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    rgb_colors = (colors * 255).astype(int)
    rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in rgb_colors]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    data=[go.Scatter3d(
        x=x,  
        y=y,  
        z=z,  
        mode='markers',  
        marker=dict(
            size=0.8,          
            color=rgb_strings,         
            opacity=0.2       
        )
    )]

    return data





def value_map(pcd_scene, current_pos=np.array([0,0,0]), target=np.array([0,0,0]), avoid=np.array([0,0,0]), show_map=True, world_trajectories=None):
    
    voxel_size = 0.01
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_scene,
                                                                voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_grid], window_name="VoxelGrid Visualization")
    max_bound = voxel_grid.get_max_bound()
    min_bound = voxel_grid.get_min_bound()

    stride_x = complex(str(int((max_bound[0] - min_bound[0]) / voxel_size)) + "j")
    stride_y = complex(str(int((max_bound[1] - min_bound[1]) / voxel_size)) + "j")
    stride_z = complex(str(int((max_bound[2] - min_bound[2]) / voxel_size)) + "j")

    X, Y, Z = np.mgrid[min_bound[0]:max_bound[0]:stride_x, min_bound[1]:max_bound[1]:stride_y, min_bound[2]:max_bound[2]:stride_z]
    values =  np.zeros(X.shape)

    x, y, z = np.meshgrid(np.arange(values.shape[0]), np.arange(values.shape[1]), np.arange(values.shape[2]))    
    indexs_scene = np.stack((x, y, z), axis=-1)
    indexs_scene = indexs_scene.reshape(-1,3)



    center_object = voxel_grid.get_voxel(target)

    distances_object = np.sqrt(np.sum(np.power((indexs_scene - center_object), 2),1))
    distance_object_max = distances_object.max()
    values_object_all = (distances_object/distance_object_max) * 6 - 3     



    sigma = 14.0  
    radius = 20
    values_avoid_all = np.zeros(len(indexs_scene))
    if len(avoid)!=0:
        for i in range(len(avoid)):
            sub_avoid = avoid[i]
            center_avoid = voxel_grid.get_voxel(sub_avoid) 
            current_index = voxel_grid.get_voxel(current_pos) 

            distance_avoid = np.sqrt(np.sum(np.power((indexs_scene - center_avoid), 2),1))
            within_mask = distance_avoid <= radius

            influence = np.exp(-np.power(distance_avoid, 2) / (2 * sigma**2)) 
            influence[~within_mask] = 0

            if np.sum(within_mask) > 0:
                min_value = np.min(influence[within_mask])
                max_value = np.max(influence[within_mask])
                if max_value > min_value:
                    influence[within_mask] = (influence[within_mask] - min_value) / (max_value - min_value) * 4

            values_avoid_all += influence 


    for index,value in enumerate(values_object_all):
        i = indexs_scene[index]
        values[i[0],i[1],i[2]] =  value + values_avoid_all[index]
        # values[i[0],i[1],i[2]] = values_avoid_all[index]

        
    return values, voxel_grid , X, Y, Z


def downsample_point_cloud(point_cloud, voxel_size=0.05):

    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_cloud


if __name__ == "__main__":
    pass
