import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets import CARTER_CFG, VELODYNE_VLP_16_RAYCASTER_CFG

from omni.isaac.lab.sensors import CameraCfg, patterns, RayCasterCfg, RayCaster, RTXRayCasterCfg, RTXRayCaster


@configclass
class CarterSceneCfg(InteractiveSceneCfg):
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # )
    environment = AssetBaseCfg(prim_path="/World/Warehouse", 
                               spawn=sim_utils.UsdFileCfg(
                                   usd_path=f"/home/social-navigation/USD-saver/full_warehouse.usd")
)
    carter: ArticulationCfg = CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        prim_path="/World/envs/env_.*/Robot/nova_carter_sensors/chassis_link/lidar",
        offset=RTXRayCasterCfg.OffsetCfg(),
        spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.EXAMPLE_SOLID_STATE),
        debug_vis=True,
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["carter"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            # reset the scene entities
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply action
        efforts = torch.tensor([[0, 60, 60, 0, 0, 0, 0]])
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        for key, value in scene["lidar"].data.items():
            for lidar_data in value:
                print(f"Data for {key}: {lidar_data.data}")
                print(f"Distance for {key}: {lidar_data.distance}")

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = CarterSceneCfg(num_envs=args_cli.num_envs, env_spacing=10.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()