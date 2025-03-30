import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm


def generate_3D_trajectory(start, end, num_traj=20, num_control_points=1, num_curve_points=100, pcd_scene=None, covariances=0.02, flat=False):

    control_points_x = np.linspace(start[0], end[0], num_control_points+2)
    control_points_y = np.linspace(start[1], end[1], num_control_points+2)
    control_points_z = np.linspace(start[2], end[2], num_control_points+2)
    key_points = np.concatenate((control_points_x[1:-1][:,np.newaxis],control_points_y[1:-1][:,np.newaxis],control_points_z[1:-1][:,np.newaxis]), axis=1)
    key_points = key_points.flatten()
    
    num_component = 2 # The number of Gaussian kernels
    gmm = GaussianMixture(n_components=num_component, covariance_type='diag')
    key_points = np.tile(key_points, (2,1))
    key_points[0,2] += 0.08
    key_points[1,2] += 0.05
    gmm.means_ = key_points
    gmm.covariances_ = np.tile(np.array([[covariances, covariances, covariances] * num_control_points]), (num_component,1))
    gmm.weights_ = np.array([0.2, 0.8]) 

    gmm_points = gmm.sample(num_traj)[0]
    gmm_points = gmm_points.reshape(num_traj,-1,3)

    trajectories = []

    for i in tqdm(range(num_traj), desc="sampling trajectories"):
        tra = gmm_points[i]
        trajectory_points = np.concatenate([start[np.newaxis, :], tra, end[np.newaxis, :]], axis=0)

        distances = np.sqrt(np.sum(np.diff(trajectory_points, axis=0) ** 2, axis=1)) 
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0) 

        uniform_distances = np.linspace(0, cumulative_distances[-1], num_curve_points)

        x, y, z = trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2]
        x_smooth = CubicSpline(cumulative_distances, x)(uniform_distances)
        y_smooth = CubicSpline(cumulative_distances, y)(uniform_distances)
        z_smooth = CubicSpline(cumulative_distances, z)(uniform_distances)

        trajectory_points = np.vstack([x_smooth, y_smooth, z_smooth]).T
        trajectory_points = redistribute_points(trajectory_points)
        trajectories.append(trajectory_points)
    
    
    trajectories = np.array(trajectories)
    if flat:
        trajectories[:,:,2] = 0.31 # 

    if pcd_scene != None:
        pcd_scene_visual = o3d.geometry.PointCloud()
        pcd_scene_visual.points = o3d.utility.Vector3dVector(np.asarray(pcd_scene.points))
        pcd_scene_visual.colors = o3d.utility.Vector3dVector(np.asarray(pcd_scene.colors))
        for i in range(num_traj):
            color = np.random.rand(3)
            trajectory_colors = np.array([color for _ in trajectories[i]])
            scene_points = np.vstack((np.asarray(pcd_scene_visual.points), trajectories[i]))
            scene_colors = np.vstack((np.asarray(pcd_scene_visual.colors), trajectory_colors))
            pcd_scene_visual.points = o3d.utility.Vector3dVector(scene_points)
            pcd_scene_visual.colors = o3d.utility.Vector3dVector(scene_colors)
        
    return np.array(trajectories), pcd_scene_visual
    
    
def redistribute_points(points):
    num_points = len(points)
    
    distances = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    
    cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_lengths[-1]
    
    target_lengths = np.linspace(0, total_length, num_points)
    
    redistributed_points = []
    for i in range(3): 
        redistributed_points.append(np.interp(target_lengths, cumulative_lengths, points[:, i]))
    
    redistributed_points = np.vstack(redistributed_points).T
    return redistributed_points

if __name__ == "__main__":
    pass