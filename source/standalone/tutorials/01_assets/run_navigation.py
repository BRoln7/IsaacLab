"""
control a wheeled robot by interaction
"""
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="control a wheeled robot on articulation")
AppLauncher.add_app_launcher_args(parser)
args_client = parser.parse_args()

app_launcher = AppLauncher(args_client)
sim_launcher = app_launcher.app

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from omni.isaac.lab_assets import CARTER_CFG

def design_scene() -> dict:
    env_cfg = sim_utils.UsdFileCfg(usd_path=f"/home/social-navigation/USD-saver/full_warehouse.usd")
    env_cfg.func("/World/full_warehouse", env_cfg)    

    # Articulation
    carter_cfg = CARTER_CFG.copy()
    carter_cfg.prim_path = "/World/Robot"
    carter = Articulation(cfg=carter_cfg)
    scene_entities = {"carter": carter}

    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    robot = entities["carter"]
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while sim_launcher.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            root_state = robot.data.default_root_state.clone()
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply action
        efforts = torch.tensor([[0, 60, 63.5, 0, 0, 0, 0]])
        # print(robot.joint_names)
        # Set the efforts to buffer
        robot.set_joint_effort_target(target=efforts)
        # write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    robot = design_scene()
    sim.reset()
    run_simulator(sim, robot)

if __name__=="__main__":
    main()
    sim_launcher.close()