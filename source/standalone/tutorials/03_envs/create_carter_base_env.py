"""
This script demonstrates how to create a simple environment with a carter. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on creating a carter base environment.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg
from omni.isaac.lab_tasks.manager_based.classic.carter.carter_env_cfg import CarterSceneCfg


@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", 
                                             joint_names=["joint_wheel_left", "joint_wheel_right"], 
                                             scale=5.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_left"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_right"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_left_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_left"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_right_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_right"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.1, 0.1),
        },
    )

@configclass
class CarterEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the carter environment."""

    # Scene settings
    scene = CarterSceneCfg(num_envs=8, env_spacing=5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        # self.viewer.eye = [4.5, 0.0, 6.0]
        # self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

def main():
    """Main function."""
    # parse the arguments
    env_cfg = CarterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("[INFO]: Resetting environment...")
            # sample random actions
            # joint_efforts = torch.randn_like(env.action_manager.action)
            joint_efforts = torch.tensor([[50, 50],
                                          [50, 50],
                                          [50, 50],
                                          [50, 50],
                                          [50, 50],
                                          [50, 50],
                                          [50, 50],
                                          [50, 50]])
            # print(env.action_manager.action)
            # print(joint_efforts)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

