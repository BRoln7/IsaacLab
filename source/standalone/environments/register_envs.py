# register a new env by gym.make

import argparse
from omni.isaac.lab.app import AppLauncher

# edit configuration of the app
# launch the app first
parser = argparse.ArgumentParser(description="Random agent for a new environment")
parser.add_argument("--disable_fabric", action="store_true", default=False, 
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, 
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, 
                    help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.manager_based.classic import humanoid
from omni.isaac.lab_tasks.utils import parse_env_cfg

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    while simulation_app.is_running():
        # run in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)
    env.close()

if __name__=="__main__":
    main()
    simulation_app.close()