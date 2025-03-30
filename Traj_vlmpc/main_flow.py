from datetime import datetime
import sys
from logger import setup_logger
import argparse
from tools import seed_all
import os

import numpy as np

from matplotlib import pyplot as plt
from tools import caculate_world_point

from flow_vlmpc import VLMPC_flow
import open3d as o3d

import time
import os

from UR5Control_mech import UR5Robot
from image2robot import image2robot_mech

parser = argparse.ArgumentParser('flow_VLMPC parameters')
parser.add_argument('--tag', type=str, )
parser.add_argument('--log_root', type=str, default='./logs')
args = parser.parse_args()


def mpc_loop(f_vlmpc, sub_goal, avoid, env, bias=np.array([0,0,-0.02]), num_exe=50, log_dir = './logs', tag="grasp", flat=False):
    print("enter loop")
    rgb_image, depth_map = env.get_mech_sensor_data()
    plt.imsave(f_vlmpc.current_obs_path, rgb_image)
    
    if sub_goal == 'pink water .':
        sub_goal_center = [1336,300]
    else:
        sub_goal_center = f_vlmpc.find_sub_goal(sub_goal,log_dir)
    end_3D = image2robot_mech([sub_goal_center[0], sub_goal_center[1]], depth_map)
    end_3D[2] += bias[2]
    avoid_centers = f_vlmpc.find_sub_goals(avoid,log_dir,1)

    avoid_3Ds = []
    for i in range(len(avoid_centers)):
        avoid_3D = image2robot_mech([avoid_centers[i][0], avoid_centers[i][1]], depth_map)
        avoid_3Ds.append(avoid_3D)

    f_vlmpc.avoid = np.array(avoid_3Ds)
    

    for num_waypoints, covariances in [(100,0.007),(50,0.001)]: 
        # TODO: Sampling trajectory
        rgb_image, depth_map = env.get_mech_sensor_data()
        plt.imsave(f_vlmpc.current_obs_path, rgb_image)
        np.save(f_vlmpc.current_depth_path, depth_map)
        cur_pos_3D = env.get_robot_state() # real end-effector is higher 0.107 than pos

        
        trajectories, pcd_trajs = f_vlmpc.generate_3D_tra(cur_pos_3D, end_3D, rgb_image, depth_map, covariances=covariances, num_traj=30, num_waypoints=num_waypoints, flat=flat)
        
        # o3d.visualization.draw_geometries([pcd_trajs], window_name="Scene with Trajectory")
        o3d.io.write_point_cloud(os.path.join(log_dir,f"{tag}_pcd_trajs{num_waypoints}.ply"), pcd_trajs)

        # TODO: cost fun
        scores, fig1, fig2 = f_vlmpc.cost_fn(trajectories, cur_pos_3D, end_3D, depth_map, num_exe, show_map=True)
        best_tra_index = scores.argmin()
        print(f"best index:{best_tra_index}")
        best_traj = trajectories[best_tra_index]

        fig1.write_html(os.path.join(log_dir,f'fig1_{sub_goal}_{num_waypoints}.html'), auto_open=False)
        fig2.write_html(os.path.join(log_dir,f'fig2_{sub_goal}_{num_waypoints}.html'), auto_open=False)

        
        # TODO: Trajectory execution
        env.move_along_path(best_traj[:num_exe])
        time.sleep(2)

    if sub_goal != "plate ." and sub_goal != 'pink water .':
        env.move_to_position_with_fixed_rotation(end_3D-bias)
    
    rgb, depth = env.get_mech_sensor_data()
    return rgb, depth, best_traj[-1]



def main(args):
    # TODO: set your robot environment
    workspace_limits = np.asarray([[0.303, 0.435], [-0.155, 0.104], [0.134, 0.291]]) 
    env = UR5Robot(tcp_host_ip='your ip address', vel=0.3, acc=0.5, camera_type='mech', workspace_limits=workspace_limits)


    # initial Traj_VLMPC
    f_vlmpc = VLMPC_flow(args)
    print("f_vlmpc init finished")

    # MPC Loop
    rgb, depth, last_pos = mpc_loop(f_vlmpc, sub_goal='peach .', avoid='brown box.', env=env, bias=np.array([0,0,0.05]), log_dir=args.log_dir, tag="grasp")
    env.close_gripper()
    rgb, depth_map = env.get_mech_sensor_data()
    plt.imsave(os.path.join(args.log_dir,"grasp.png"), rgb)

    time.sleep(2)
    rgb, depth, last_pos = mpc_loop(f_vlmpc, sub_goal='plate .', avoid='brown box.',  env=env, bias=np.array([0,0,0.1]), log_dir=args.log_dir, tag="place", flat=False)
    env.open_gripper()
    
    rgb, depth_map = env.get_mech_sensor_data()
    plt.imsave(os.path.join(args.log_dir,"place.png"), rgb)



if __name__ == "__main__":
    

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = os.path.join(args.log_root, '%s_%s'%(args.tag, stamp))
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(output=args.log_dir, color=True, name='Flow_vlmpc')
    main(args)
    
