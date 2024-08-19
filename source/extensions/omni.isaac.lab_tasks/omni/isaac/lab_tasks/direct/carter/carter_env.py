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
from omni.isaac.lab.sensors import CameraCfg


@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 100.0  # [N]
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


class CarterEnv(DirectRLEnv):
    cfg: CarterEnvCfg

    def __init__(self, cfg: CarterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_dof_idx, _ = self.carter.find_joints(self.cfg.left_dof_name)
        self._right_dof_idx, _ = self.carter.find_joints(self.cfg.right_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.carter.data.joint_pos
        self.joint_vel = self.carter.data.joint_vel

    def _setup_scene(self):
        env_cfg = sim_utils.UsdFileCfg(usd_path=f"/home/social-navigation/USD-saver/full_warehouse.usd")
        env_cfg.func("/World/full_warehouse", env_cfg)  
        self.carter = Articulation(self.cfg.robot_cfg)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["carter"] = self.carter


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        print("self.action_scale:", self.action_scale)
        print("self.actions", self.actions)

    def _apply_action(self) -> None:
        self.carter.set_joint_effort_target(target=self.actions)

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._left_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._left_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._right_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._right_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        obs = torch.cat((self.ped_pos, self.scan, self.goal), axis=None)

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
