# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence

from omni.isaac.lab_assets.carter import CARTER_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import CameraCfg, RTXRayCasterCfg, RTXRayCaster
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

NUM_TP = 10
device = "cuda:0"
motor_scale = 0.5   # corresponding to joint 42 power
joint_base = 42
base_width = 0.4132
system_gain = 1.25

# visualize
# lidar marker
marker_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/LidarMarkers",
    markers={
        "marker": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
lidar_marker = VisualizationMarkers(marker_cfg)
# goal marker
goal_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/GoalMarkers",
    markers={
        "marker": sim_utils.ConeCfg(
            radius=0.6,
            height=0.8,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.5)),
        ),
    }
)
goal_marker = VisualizationMarkers(goal_cfg)

goal = torch.Tensor([[-5.0, -2.0],
                     [-10.0, -2.0]]).to(device)

@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 512.0
    action_scale = motor_scale  # [N]
    num_actions = 2
    num_observations = 19202 
    num_states = 0
    

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    left_dof_name = "joint_wheel_left"
    right_dof_name = "joint_wheel_right"

    # sensors
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/nova_carter_sensors/chassis_link/front_hawk/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    lidar = RTXRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/nova_carter_sensors/chassis_link/front_RPLidar/lidar",
        offset=RTXRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.VELODYNE_VLS128),
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=5.0, replicate_physics=True)

    # reset
    max_cart_pos = 40.0  # the carter is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

"""
for visualize
"""
def quat_to_rot_matrix(quat):
    rot_matrix = []
    for i in range(quat.size(0)):
        quat_tmp = quat[i]
        q = quat_tmp / torch.norm(quat_tmp)
        w, x, y, z = q
        rot_matrix_tmp = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], device=device)
        rot_matrix.append(rot_matrix_tmp)
    rot_matrix = torch.stack(rot_matrix)
    return rot_matrix

def transform_to_robot(data, rot_matrix, translation):
    # print("data shape:", data.shape, rot_matrix.shape)
    data_transformed = []
    for i in range(data.size(0)):
        data_tmp = data[i]
        rot_matrix_tmp = rot_matrix[i]
        translation_tmp = translation[i]
        data_rotated = torch.mm(data_tmp.to(device), rot_matrix_tmp.to(device).t())
        data_transformed_tmp = data_rotated + translation_tmp
        data_transformed.append(data_transformed_tmp)
    data_transformed = torch.stack(data_transformed)
    # print("data_transformed shape:", data_transformed.shape)
    return data_transformed

def quaternion_to_angle(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r11 = 1 - 2 * (y**2 + z**2)
    r21 = 2 * (x * y - w * z)
    angles = torch.atan2(r21, r11)
    angles = (angles + math.pi) % (2 * math.pi) - math.pi
    return angles

def goal_rotate(angle, relative_goal):
    x_new = relative_goal[0] * torch.cos(angle) - relative_goal[1] * torch.sin(angle)
    y_new = relative_goal[0] * torch.sin(angle) + relative_goal[1] * torch.cos(angle)
    relative_goal[0] = x_new
    relative_goal[1] = y_new
    return relative_goal

class CarterEnv(DirectRLEnv):
    cfg: CarterEnvCfg

    def __init__(self, cfg: CarterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_dof_idx, _ = self.carter.find_joints(self.cfg.left_dof_name)
        self._right_dof_idx, _ = self.carter.find_joints(self.cfg.right_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.carter.data.joint_pos
        self.joint_vel = self.carter.data.joint_vel

        self.lidar_history_data = torch.Tensor([[]]).to(device)
        self.pooling_tns = 0


    def _setup_scene(self):
        env_cfg = sim_utils.UsdFileCfg(usd_path=f"/home/social-navigation/USD-saver/full_warehouse.usd")
        env_cfg.func("/World/full_warehouse", env_cfg)  
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.carter = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["carter"] = self.carter
        self.lidar = RTXRayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self.lidar

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        # print("self.action_scale:", self.action_scale)
        print("self.actions", self.actions)

    def _apply_action(self) -> None:
        controller = differential_drive_solver(self.actions)
        # controller = torch.zeros(self.num_envs, 7).to(device)
        self.carter.set_joint_effort_target(target=controller)

    def _get_observations(self) -> dict:
        # process lidar_data
        lidar_data = torch.Tensor([]).to(device)
        lidar_ranges = torch.Tensor([]).to(device)
        lidar_data, lidar_ranges = raw_lidar_process(self.lidar)
        lidar_data_final, pooling_flag, self.lidar_history_data, self.pooling_tns =\
              lidar_pooling(self.lidar_history_data, lidar_ranges, self.pooling_tns)
        
        # visulize lidar data
        if lidar_data.dim() > 1 and lidar_data.size(1) >= 300:
            lidar_to_robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0]]).to(device)
            lidar_to_robot_pos = torch.tensor([[0.026, 0.0, 0.418],
                                               [0.026, 0.0, 0.418]]).to(device)
            rot_matrix = quat_to_rot_matrix(lidar_to_robot_quat)
            lidar_robot_data = transform_to_robot(lidar_data, rot_matrix, lidar_to_robot_pos)

            robot_to_world_quat = self.carter.data.root_quat_w
            robot_to_world_pos = self.carter.data.root_pos_w
            robot_rot_matrix = quat_to_rot_matrix(robot_to_world_quat)
            lidar_data_world = transform_to_robot(lidar_robot_data, robot_rot_matrix, robot_to_world_pos)

            lidar_data_world = lidar_data_world.reshape(-1, 3)
            lidar_marker.visualize(translations=lidar_data_world)

        # process peds data 
        ped_pos = torch.zeros(self.num_envs, 12800).to(device)    

        # process subgoal data
        robot_to_world_quat = self.carter.data.root_quat_w
        robot_to_world_pos = self.carter.data.root_pos_w[:, :2]
        relative_goal = goal - robot_to_world_pos
        angle_x_axis = quaternion_to_angle(robot_to_world_quat)
        print("angle_x_axis:", angle_x_axis)
        for i in range(relative_goal.size(0)):
            relative_goal[i] = goal_rotate(angle_x_axis[i], relative_goal[i])
        print("goal_in_robot:", relative_goal)
        lookahead = torch.Tensor([2.0]).to(device)
        subgoal = subgoal_get(relative_goal, lookahead)

        z_zeros = torch.zeros(goal.shape[0], 1).to(device)
        goal_with_z = torch.cat((goal, z_zeros), dim=-1)  # Concatenate along the last dimension to add the z-axis
        goal_marker.visualize(translations=goal_with_z.reshape(self.num_envs, 3))


        # concat observation
        if pooling_flag:
            # print("peds, lidar and subgoal shape:",ped_pos.shape, lidar_data_final.shape, subgoal.shape)
            obs = torch.cat((ped_pos, lidar_data_final, subgoal), dim=1)
        else:
            obs = torch.zeros(self.num_envs, 19202).to(device)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._left_dof_idx[0]],
            self.joint_vel[:, self._left_dof_idx[0]],
            self.joint_pos[:, self._right_dof_idx[0]],
            self.joint_vel[:, self._right_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.carter.data.joint_pos
        self.joint_vel = self.carter.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if torch.any(time_out):
            print("TIME OUT!!", time_out)
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._left_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        if torch.any(out_of_bounds):
            print("OUT OF BOUND:", out_of_bounds)    
        # print(out_of_bounds)    
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carter._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.carter.data.default_joint_pos[env_ids]
        joint_pos[:, self._left_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._left_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.carter.data.default_joint_vel[env_ids]

        default_root_state = self.carter.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.carter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.carter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.carter.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward


def raw_lidar_process(_lidar:RTXRayCaster) -> tuple[torch.Tensor, torch.Tensor]:
    """
    load raw lidar data and extract the 720 points needed
    """
    lidar = _lidar
    lidar_data = lidar.data[list(lidar.data.keys())[0]][0].data
    # print("lidar_data size:",lidar_data.shape)
    if lidar_data.size(0) != 0:
        lidar_data_list = []
        lidar_ranges_list = []
        for key in lidar.data:
            lidar_data = lidar.data[key][0].data
            lidar_ranges = lidar.data[key][0].distance
            z_min = -0.02
            z_max = 0.02
            z_data = lidar_data[:, 2]
            mask = (z_data >= z_min) & (z_data <= z_max)
            filtered_lidar_data = lidar_data[mask][0:970]
            filtered_lidar_ranges = lidar_ranges[mask][0:970]
            num_samples = 720
            indices = torch.linspace(0, filtered_lidar_data.size(0) - 1, num_samples).long()
            filtered_lidar_data = filtered_lidar_data[indices]
            filtered_lidar_ranges = filtered_lidar_ranges[indices]    
            # print("filtered lidar_data size:", filtered_lidar_data.shape)
            lidar_data_list.append(filtered_lidar_data)
            lidar_ranges_list.append(filtered_lidar_ranges)
        final_lidar_data = torch.stack(lidar_data_list)
        final_lidar_ranges = torch.stack(lidar_ranges_list)
        # print("final_lidar_data shape:", final_lidar_data.shape, 
        #       "final_lidar_range shape", final_lidar_ranges.shape)
        return final_lidar_data.to(device), final_lidar_ranges.to(device)
    else: 
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)

def lidar_pooling(_lidar_history_data:torch.Tensor, 
                   _scan_tmp:torch.Tensor, 
                   _pooling_tns)->tuple[torch.Tensor, bool, torch.Tensor, int]:
    """
    MAX-AVG Pooling
    _lidar_history_data: history frame data
    _scan_tmp: current frame data
    _pooling_tns: times poolinged
    cnn_lidar: full data to be processed
    """
    if _scan_tmp.dim() <= 1:
        return torch.Tensor([]).to(device), False, torch.Tensor([]).to(device), 0

    # print("_lidar_history_data.shape:", _lidar_history_data.shape)
    # print("_scan_tmp.shape:", _scan_tmp.shape)
    
    # stack current frame scan 
    cnn_lidar_tmp = []
    for i in range(_scan_tmp.size(0)):
        if _lidar_history_data.size(0) == 0:
            cnn_lidar_tmp.append(_scan_tmp[i].reshape(1, 720))
        else:
            cnn_lidar_tmp.append(torch.cat((_lidar_history_data[i], 
                                            _scan_tmp[i].reshape(1, 720)), dim=0))
    _pooling_tns += 1
    cnn_lidar_tmp = torch.stack(cnn_lidar_tmp)
    # print("cnn_lidar_tmp shape:", cnn_lidar_tmp.shape)
    
    if _pooling_tns == NUM_TP:
        cnn_lidar = cnn_lidar_tmp.reshape(_scan_tmp.size(0), -1)
        # print("cnn_lidar shape:", cnn_lidar.shape)
        # Kick out frame 0 lidar data
        _pooling_tns = NUM_TP - 1
        _lidar_history_data = cnn_lidar_tmp[:, 1:NUM_TP]
        # print("lidar_history_data shape:", _lidar_history_data.shape)
        
        # MaxAbsScaler:
        lidar_avg_all = []
        for j in range(_scan_tmp.size(0)):
            lidar_avg = torch.zeros((20, 80)).to(device)
            for n in range(NUM_TP):
                lidar_tmp = cnn_lidar[j, n*720:(n+1)*720]
                # if n == 9:
                #     print("lidar tmp:",lidar_tmp)
                for i in range(80):
                    lidar_avg[2*n, i] = torch.min(lidar_tmp[i*9:(i+1)*9])
                    lidar_avg[2*n+1, i] = torch.mean(lidar_tmp[i*9:(i+1)*9])
            lidar_avg_all.append(lidar_avg)
        lidar_avg_all = torch.stack(lidar_avg_all)
        # print("lidar_avg_all shape:", lidar_avg_all.shape)
        
        # stack 4 times
        lidar_data_final = []
        for lidar_avg in lidar_avg_all:
            lidar_avg = lidar_avg.reshape(1600)
            lidar_avg_map = lidar_avg.repeat(1, 4)
            lidar_avg_map = lidar_avg_map.reshape(6400)
            s_min = 0
            s_max = 30
            lidar_avg_map = 2 * (lidar_avg_map - s_min) / (s_max - s_min) + (-1)
            lidar_data_final.append(lidar_avg_map)
        lidar_data_final = torch.stack(lidar_data_final)
        # print("lidar_data_final shape:",lidar_data_final.shape)
        return lidar_data_final, True, _lidar_history_data, _pooling_tns

    else:
        return cnn_lidar_tmp, False, cnn_lidar_tmp, _pooling_tns

def ped_encoder():
    """
    encode peds' poses and positions into grid maps in world-frame
    """
    # # get the pedstrain's position:
    # self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
    # if(len(trackPed_msg.tracks) != 0):  # tracker results
    #     for ped in trackPed_msg.tracks:
    #         #ped_id = ped.track_id 
    #         # create pedestrian's postion costmap: 10*10 m
    #         x = ped.pose.pose.position.x
    #         y = ped.pose.pose.position.y
    #         vx = ped.twist.twist.linear.x
    #         vy = ped.twist.twist.linear.y
    #         # 20m * 20m occupancy map:
    #         if(x >= 0 and x <= 20 and np.abs(y) <= 10):
    #             # bin size: 0.25 m
    #             c = int(np.floor(-(y-10)/0.25))
    #             r = int(np.floor(x/0.25))

    #             if(r == 80):
    #                 r = r - 1
    #             if(c == 80):
    #                 c = c - 1
    #             # cartesian velocity map
    #             self.ped_pos_map_tmp[0,r,c] = vx
    #             self.ped_pos_map_tmp[1,r,c] = vy

def ped_transformer():
    """
    transform the peds' grid maps from world frame into robot frame
    """

def subgoal_get(_goal:torch.Tensor, _lookahead:torch.Tensor)->torch.Tensor:
    subgoal_in_robot = []
    g_min = -2
    g_max = 2
    for i in range(_goal.size(0)):
        goal_in_robot_tmp = torch.Tensor([_goal[i, 0], _goal[i, 1]]).to(device)
        if torch.sqrt(goal_in_robot_tmp[0] * goal_in_robot_tmp[0]+
                      goal_in_robot_tmp[1] * goal_in_robot_tmp[1]) > _lookahead:
            scale_factor = _lookahead / torch.sqrt(goal_in_robot_tmp[0] * goal_in_robot_tmp[0]+
                                                   goal_in_robot_tmp[1] * goal_in_robot_tmp[1])
            subgoal_in_robot_tmp = torch.Tensor([goal_in_robot_tmp[0]*scale_factor,
                                                 goal_in_robot_tmp[1]*scale_factor]).to(device)
        else:
            subgoal_in_robot_tmp = goal_in_robot_tmp
        # print("subgoal_in_robot:", subgoal_in_robot_tmp)
        subgoal_in_robot_tmp = 2 * (subgoal_in_robot_tmp - g_min) / (g_max - g_min) + (-1)
        subgoal_in_robot.append(subgoal_in_robot_tmp)
    subgoal_in_robot = torch.stack(subgoal_in_robot)

    return subgoal_in_robot


def differential_drive_solver(actions:torch.Tensor)->torch.Tensor:
    action_list = []
    for i in range(actions.size(0)):
        vx = actions[i, 0]
        wf = actions[i, 1]
        action = torch.zeros(7).to(device)

        vx_base = motor_scale 
        pl_pr_base = joint_base  
        gain = system_gain  # Adjust gain based on the system's response

        v_left = vx - (wf * base_width / 2.0)
        v_right = vx + (wf * base_width / 2.0)

        pl = gain * (v_left / vx_base) * pl_pr_base
        pr = gain * (v_right / vx_base) * pl_pr_base

        action[1] = pl
        action[2] = pr

        action_list.append(action)
    action_list = torch.stack(action_list)
    # print("action:", action_list)
    return action_list