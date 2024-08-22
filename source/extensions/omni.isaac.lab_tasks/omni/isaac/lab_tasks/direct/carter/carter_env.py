# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
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
motor_scale = 50

# visualize
marker_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/LidarMarkers",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
lidar_marker = VisualizationMarkers(marker_cfg)

@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=5.0, replicate_physics=True)

    # reset
    max_cart_pos = 5.0  # the carter is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


def quat_to_rot_matrix(quat):
    q = quat / torch.norm(quat)
    w, x, y, z = q
    rot_matrix = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], device=device)
    return rot_matrix

def transform_to_robot(data, rot_matrix, translation):
    data_rotated = torch.mm(data.to(device), rot_matrix.to(device).t())
    data_transformed = data_rotated + translation
    return data_transformed

class CarterEnv(DirectRLEnv):
    cfg: CarterEnvCfg

    def __init__(self, cfg: CarterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_dof_idx, _ = self.carter.find_joints(self.cfg.left_dof_name)
        self._right_dof_idx, _ = self.carter.find_joints(self.cfg.right_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.carter.data.joint_pos
        self.joint_vel = self.carter.data.joint_vel

        self.lidar_history_data = torch.Tensor([]).to(device)
        self.pooling_tns = 0

    def _setup_scene(self):
        env_cfg = sim_utils.UsdFileCfg(usd_path=f"/home/social-navigation/USD-saver/full_warehouse.usd")
        env_cfg.func("/World/full_warehouse", env_cfg)  
        self.carter = Articulation(self.cfg.robot_cfg)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["carter"] = self.carter
        self.lidar = RTXRayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self.lidar

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        print("self.action_scale:", self.action_scale)
        print("self.actions", self.actions)

    def _apply_action(self) -> None:
        controller = torch.zeros(2, 7).to(device)
        # self.carter.set_joint_effort_target(target=self.actions)
        self.carter.set_joint_effort_target(target=controller)

    def _get_observations(self) -> dict:
        # process lidar_data
        lidar_data = torch.Tensor([]).to(device)
        lidar_ranges = torch.Tensor([]).to(device)
        lidar_data, lidar_ranges = raw_lidar_process(self.lidar)
        lidar_data_final, pooling_flag, self.lidar_history_data, self.pooling_tns =\
              _lidar_Pooling(self.lidar_history_data, lidar_ranges, self.pooling_tns)
        # visulize lidar
        if lidar_ranges.size(0) >= 300:
            lidar_to_robot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(device)
            lidar_to_robot_pos = torch.tensor([0.026, 0.0, 0.418]).to(device)
            rot_matrix = quat_to_rot_matrix(lidar_to_robot_quat)
            lidar_robot_data = transform_to_robot(lidar_ranges, rot_matrix, lidar_to_robot_pos)

            robot_to_world_quat = self.carter.data.root_quat_w[0]
            robot_to_world_pos = self.carter.data.root_pos_w[0]
            robot_rot_matrix = quat_to_rot_matrix(robot_to_world_quat)
            lidar_data_world = transform_to_robot(lidar_robot_data, robot_rot_matrix, robot_to_world_pos)

            lidar_marker.visualize(translations=lidar_data_world)

        # process peds data 

        # process subgoal data


        # concat observation
        # if pooling_flag:
        #     obs = torch.cat((self.ped_pos, lidar_data_final, self.goal), axis=None)
        obs = torch.zeros(19202).to(device) / 50

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
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._left_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._right_dof_idx]) > math.pi / 2, dim=1)
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
    # print("lidar:"type(lidar))
    lidar_data = lidar.data[list(lidar.data.keys())[0]][0].data
    lidar_ranges = lidar.data[list(lidar.data.keys())[0]][0].distance
    print("lidar_data size:",lidar_data.shape)
    if lidar_data.size(0) != 0:
        print("lidar_data size:", lidar_data)
        z_min = -0.02
        z_max = 0.02
        z_data = lidar_data[:, 2]
        mask = (z_data >= z_min) & (z_data <= z_max)
        filtered_lidar_data = lidar_data[mask][0:950]
        filtered_lidar_ranges = lidar_ranges[mask][0:950]
        num_samples = 720
        indices = torch.linspace(0, filtered_lidar_data.size(0) - 1, num_samples).long()
        filtered_lidar_data = filtered_lidar_data[indices]
        filtered_lidar_ranges = filtered_lidar_ranges[indices]    
        print(filtered_lidar_data.shape)
        return filtered_lidar_data, filtered_lidar_ranges
    else: 
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)

def _lidar_Pooling(_lidar_history_data:torch.Tensor, 
                   _scan_tmp:torch.Tensor, 
                   _pooling_tns)->tuple[torch.Tensor, bool, torch.Tensor, int]:
    """
    MAX-AVG Pooling
    _lidar_history_data: history frame data
    _scan_tmp: current frame data
    _pooling_tns: times poolinged
    cnn_lidar: full data to be processed
    """
    cnn_lidar_tmp = torch.cat((_lidar_history_data, _scan_tmp), dim=0)
    _pooling_tns += 1
    
    if _pooling_tns == NUM_TP:
        cnn_lidar = cnn_lidar_tmp.flatten()
        # Kick out frame 0 lidar data
        _pooling_tns = NUM_TP - 1
        _lidar_history_data = cnn_lidar_tmp[1:NUM_TP]
        # MaxAbsScaler:
        lidar_avg = torch.zeros((20,80)).to(device)
        for n in range(NUM_TP):
            lidar_tmp = cnn_lidar[n*720:(n+1)*720]
            for i in range(80):
                lidar_avg[2*n, i] = torch.min(lidar_tmp[i*9:(i+1)*9])
                lidar_avg[2*n+1, i] = torch.mean(lidar_tmp[i*9:(i+1)*9])
        # stack 4 times
        lidar_avg = lidar_avg.reshape(1600)
        lidar_avg_map = lidar_avg.repeat(1, 4)
        lidar_data_final = lidar_avg_map.reshape(6400)
        s_min = 0
        s_max = 30
        lidar_data_final = 2 * (lidar_data_final - s_min) / (s_max - s_min) + (-1)

        return lidar_data_final, True, _lidar_history_data, _pooling_tns

    else:
        return cnn_lidar_tmp, False, cnn_lidar_tmp, _pooling_tns

def Different_Controller() -> tuple[torch.Tensor]:
    Controller = torch.zeros()