import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
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

from omni.isaac.lab_assets import CARTER_CFG

from omni.isaac.lab.sensors import CameraCfg, RTXRayCasterCfg, RTXRayCaster

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

import numpy as np

@configclass
class CarterSceneCfg(InteractiveSceneCfg):
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
        prim_path="/World/envs/env_.*/Robot/nova_carter_sensors/chassis_link/front_RPLidar/lidar",
        offset=RTXRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.VELODYNE_VLS128),
        debug_vis=True,
    )

marker_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/testMarkers",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
lidar_marker = VisualizationMarkers(marker_cfg)

def lidar_process(scene: InteractiveScene)->tuple[torch.Tensor, torch.Tensor]:
    lidar = scene["lidar"]
    print(type(lidar))
    lidar_data = lidar.data[list(lidar.data.keys())[0]][0].data
    lidar_ranges = lidar.data[list(lidar.data.keys())[0]][0].distance
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

def quat_to_rot_matrix(quat):
    q = quat / torch.norm(quat)
    w, x, y, z = q
    rot_matrix = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], device="cuda:0")
    return rot_matrix

def transform_to_robot(data, rot_matrix, translation):
    data_rotated = torch.mm(data.to("cuda:0"), rot_matrix.to("cuda:0").t())
    data_transformed = data_rotated + translation
    return data_transformed

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

        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        # print("-------------------------------")
        filtered_lidar_data, filtered_lidar_ranges = lidar_process(scene)
        """
        for visualize with markers
        """
        if filtered_lidar_data.size(0) >= 300:
            lidar_to_robot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
            lidar_to_robot_pos = torch.tensor([0.026, 0.0, 0.418], device="cuda:0")
            rot_matrix = quat_to_rot_matrix(lidar_to_robot_quat)
            robot_data = transform_to_robot(filtered_lidar_data, rot_matrix, lidar_to_robot_pos)

            robot_to_world_quat = robot.data.root_quat_w[0]
            robot_to_world_pos = robot.data.root_pos_w[0]
            robot_rot_matrix = quat_to_rot_matrix(robot_to_world_quat)
            lidar_data_world = transform_to_robot(robot_data, robot_rot_matrix, robot_to_world_pos)

            lidar_marker.visualize(translations=lidar_data_world)


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