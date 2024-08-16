# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors.imu import ImuCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.markers.config import RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.utils.assets import NUCLEUS_ASSET_ROOT_DIR  # isort: skip

# offset of imu_link from base on the Anymal C
# POS_OFFSET = (0.2488, 0.00835, 0.04628)
# ROT_OFFSET = (0.7071068, 0, 0, 0.7071068)

# offset of imu_link from link_1 on simple_2_link
POS_OFFSET = (0.4, 0.0, 0.1)
ROT_OFFSET = (0.5, 0.5, 0.5, 0.5)

def vector_error(v1: torch.Tensor, v2: torch.Tensor):
    """Returns the magnitude and direction error between two vectors"""
    v1_mag = torch.linalg.vector_norm(v1,dim=-1)
    v2_mag = torch.linalg.vector_norm(v2,dim=-1)
    v1_dir = torch.transpose(torch.div(torch.transpose(v1,0,-1),v1_mag),0,-1)
    v2_dir = torch.transpose(torch.div(torch.transpose(v2,0,-1),v2_mag),0,-1)
    mag_abs_err = torch.abs(v1_mag-v2_mag)
    mag_rel_err = torch.abs(v1_mag/v2_mag-1.0)
    dir_err= torch.acos(torch.sum(v1_dir*v2_dir,dim=-1))

    return mag_abs_err, mag_rel_err, dir_err

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     max_init_terrain_level=None,
    # )

    # rigid objects - balls
    balls = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.126)),
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, 0.126)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    # articulations - robot
    # robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=sim_utils.UrdfFileCfg(
                fix_base=True,
                merge_fixed_joints=False,
                make_instanceable=False,
                asset_path=f"{os.path.abspath(os.getcwd())}/urdfs/simple_2_link.urdf",
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0
                )
            ),
            init_state=ArticulationCfg.InitialStateCfg(),
            actuators={
                "joint_1_act" : ImplicitActuatorCfg(joint_names_expr=["joint_.*"],
                                          stiffness=0.0,
                                          damping=0.5),
            },
            
                
    )
    # sensors - imu (filled inside unit test)
    imu_ball: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/ball",
    )
    imu_cube: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/cube",
    )
    imu_robot_imu_link: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/imu_link",
        debug_vis=not app_launcher._headless,
        visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/imu_link")

    )
    imu_robot_base: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/link_1",
        offset=ImuCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
        debug_vis=not app_launcher._headless,
        visualizer_cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/base")
    )

    
    def __post_init__(self):
        """Post initialization."""
        # change position of the robot
        self.robot.init_state.pos = (0.0, 2.0, 0.0)
        self.imu_robot_imu_link.visualizer_cfg.markers["arrow"].scale = (1.0, 0.05, 0.05)
        self.imu_robot_base.visualizer_cfg.markers["arrow"].scale = (1.0, 0.05, 0.05)
        # self.imu_robot_base.visualizer_cfg.prim_path = "/Visuals/Command/acc_base"
        # change asset
        # self.robot.spawn.usd_path = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Robots/ANYbotics/anymal_c.usd"
        # change iterations
        # self.robot.spawn.articulation_props.solver_position_iteration_count = 32
        # self.robot.spawn.articulation_props.solver_velocity_iteration_count = 32


class TestImu(unittest.TestCase):
    """Test for Imu sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load simulation context
        sim_cfg = sim_utils.SimulationCfg(dt=0.005,device="cpu",use_fabric=False)
        sim_cfg.physx.solver_type = 0  # 0: PGS, 1: TGS --> use PGS for more accurate results
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # construct scene
        scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
        self.scene = InteractiveScene(scene_cfg)
        # Play the simulator
        self.sim.reset()

    def tearDown(self):
        """Stops simulator after each test."""
        # clear the stage
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_constant_velocity(self):
        """Test the Imu sensor with a constant velocity.

        Expected behavior is that the linear and angular are approx the same at every time step as in each step we set
        the same velocity and therefore reset the physx buffers."""
        prev_lin_acc_ball = torch.zeros((self.scene.num_envs, 3), dtype=torch.float32, device=self.scene.device)
        prev_ang_acc_ball = torch.zeros((self.scene.num_envs, 3), dtype=torch.float32, device=self.scene.device)
        prev_lin_acc_cube = torch.zeros((self.scene.num_envs, 3), dtype=torch.float32, device=self.scene.device)
        prev_ang_acc_cube = torch.zeros((self.scene.num_envs, 3), dtype=torch.float32, device=self.scene.device)

        # turn off gravity
        self.sim.gravity=(0.0,0.0,0.0)

        for idx in range(100):
            # set velocity
            self.scene.rigid_objects["balls"].write_root_velocity_to_sim(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                    self.scene.num_envs, 1
                )
            )
            self.scene.rigid_objects["cube"].write_root_velocity_to_sim(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                    self.scene.num_envs, 1
                )
            )
            # write data to sim
            self.scene.write_data_to_sim()

            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

            # print("ball linear acc: ",self.scene.sensors["imu_ball"].data.lin_acc_b)
            # print("ball linear acc prev: ",prev_lin_acc_ball)
            # print("ball linear vel: ",self.scene.sensors["imu_ball"].data.lin_vel_b)

            if idx > 1:
                # check the imu accelerations
                torch.testing.assert_close(
                    self.scene.sensors["imu_ball"].data.lin_acc_b,
                    prev_lin_acc_ball,
                    rtol=1e-3,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    self.scene.sensors["imu_ball"].data.ang_acc_b,
                    prev_ang_acc_ball,
                    rtol=1e-3,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    self.scene.sensors["imu_cube"].data.lin_acc_b,
                    prev_lin_acc_cube,
                    rtol=1e-3,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    self.scene.sensors["imu_cube"].data.ang_acc_b,
                    prev_ang_acc_cube,
                    rtol=1e-3,
                    atol=1e-3,
                )

                # check the imu velocities
                torch.testing.assert_close(
                    self.scene.sensors["imu_ball"].data.lin_vel_b,
                    torch.tensor([[1.0, 1e-6, -self.scene.physics_dt*9.81]], dtype=torch.float32, device=self.scene.device).repeat(
                        self.scene.num_envs, 1
                    )
                )
                torch.testing.assert_close(
                    self.scene.sensors["imu_cube"].data.lin_vel_b,
                    torch.tensor([[1.0, 1e-6, -self.scene.physics_dt*9.81]], dtype=torch.float32, device=self.scene.device).repeat(
                        self.scene.num_envs, 1
                    )
                )

            # update previous values
            prev_lin_acc_ball = self.scene.sensors["imu_ball"].data.lin_acc_b.clone()
            prev_ang_acc_ball = self.scene.sensors["imu_ball"].data.ang_acc_b.clone()
            prev_lin_acc_cube = self.scene.sensors["imu_cube"].data.lin_acc_b.clone()
            prev_ang_acc_cube = self.scene.sensors["imu_cube"].data.ang_acc_b.clone()

    # def test_constant_acceleration(self):
    #     """Test the Imu sensor with a constant acceleration."""
    #     for idx in range(10):
    #         # set acceleration
    #         self.scene.rigid_objects["balls"].write_root_velocity_to_sim(
    #             torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
    #                 self.scene.num_envs, 1
    #             )
    #             * (idx + 1)
    #         )
    #         # write data to sim
    #         self.scene.write_data_to_sim()
    #         # perform step
    #         self.sim.step()
    #         # read data from sim
    #         self.scene.update(self.sim.get_physics_dt())

    #         # skip first step where initial velocity is zero
    #         if idx < 1:
    #             continue

    #         # check the imu data
    #         torch.testing.assert_close(
    #             self.scene.sensors["imu_ball"].data.lin_acc_b,
    #             math_utils.quat_rotate_inverse(
    #                 self.scene.rigid_objects["balls"].data.root_quat_w,
    #                 torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
    #                     self.scene.num_envs, 1
    #                 )
    #                 / self.sim.get_physics_dt(),
    #             ),
    #             rtol=1e-4,
    #             atol=1e-4,
    #         )

    #         # check the angular velocity
    #         torch.testing.assert_close(
    #             self.scene.sensors["imu_ball"].data.ang_vel_b,
    #             self.scene.rigid_objects["balls"].data.root_ang_vel_b,
    #             rtol=1e-4,
    #             atol=1e-4,
    #         )

    def test_offset_calculation(self):
        # should achieve same results between the two imu sensors on the robot
        for idx in range(1000):
            # set acceleration
            # self.scene.articulations["robot"].write_root_velocity_to_sim(
            #     torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
            #         self.scene.num_envs, 1
            #     )
            #     * (idx + 1)
            # )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

             # get robot joint state
            joint_pos = self.scene.articulations["robot"].data.joint_pos
            joint_vel_state = self.scene.articulations["robot"].data.joint_vel
            joint_acc_state = self.scene.articulations["robot"].data.joint_acc

            lin_vel_w_imu_link = math_utils.quat_rotate(self.scene.sensors["imu_robot_imu_link"].data.quat_w,self.scene.sensors["imu_robot_imu_link"].data.lin_vel_b)
            lin_vel_w_base = math_utils.quat_rotate(self.scene.sensors["imu_robot_base"].data.quat_w,self.scene.sensors["imu_robot_base"].data.lin_vel_b)
            lin_acc_w_imu_link = math_utils.quat_rotate(self.scene.sensors["imu_robot_imu_link"].data.quat_w,self.scene.sensors["imu_robot_imu_link"].data.lin_acc_b)
            lin_acc_w_base = math_utils.quat_rotate(self.scene.sensors["imu_robot_base"].data.quat_w,self.scene.sensors["imu_robot_base"].data.lin_acc_b)

            joint_vel_imu = math_utils.quat_rotate(self.scene.sensors["imu_robot_imu_link"].data.quat_w,self.scene.sensors["imu_robot_imu_link"].data.ang_vel_b)[...,1].unsqueeze(-1)
            joint_acc_imu = math_utils.quat_rotate(self.scene.sensors["imu_robot_imu_link"].data.quat_w,self.scene.sensors["imu_robot_imu_link"].data.ang_acc_b)[...,1].unsqueeze(-1)
            
            # ang_vel_w_base= math_utils.quat_rotate(self.scene.sensors["imu_robot_imu_link"].data.quat_w,self.scene.sensors["imu_robot_imu_link"].data.ang_vel_b)
            
            print()
            print(idx)
            # print("joint_pos: ", joint_pos)
            print("joint_vel_error: ", joint_vel_imu-joint_vel_state)
            print("joint_acc_error: ", joint_acc_imu-joint_acc_state)
            # print("offset_base: ", self.scene.sensors["imu_robot_base"]._offset_pos_b)
            # print("offset_imu_link: ", self.scene.sensors["imu_robot_imu_link"]._offset_pos_b)
            # # print("GT linear velocity wrt world")
            vx = -joint_vel_imu * POS_OFFSET[0]*torch.sin(joint_pos)
            vy = torch.zeros(2,1,device=self.scene.device)
            vz = -joint_vel_imu* POS_OFFSET[0]*torch.cos(joint_pos)
            gt_linear_vel_w = torch.cat([vx,vy,vz],dim=-1)

            print("linear velocity")
            print("GT: ",gt_linear_vel_w)
            print("GT_error_base: ",vector_error(gt_linear_vel_w,lin_vel_w_base))
            print("GT_error_imu_link: ",vector_error(gt_linear_vel_w,lin_vel_w_imu_link))
            print("compare_error: ",vector_error(lin_vel_w_base,lin_vel_w_imu_link))
            # print("base: ",lin_vel_w_base)
            # print("imu_link: ",lin_vel_w_imu_link)

            # print("GT Linear acceleration wrt world")
            ax = -joint_acc_imu * POS_OFFSET[0]*torch.sin(joint_pos) - joint_vel_imu**2 * POS_OFFSET[0] *torch.cos(joint_pos)
            ay = torch.zeros(2,1,device=self.scene.device)
            az = -joint_acc_imu * POS_OFFSET[0]*torch.cos(joint_pos) - joint_vel_imu**2 * POS_OFFSET[0] *torch.sin(joint_pos) + 9.81
            gt_linear_acc_w = torch.cat([ax,ay,az],dim=-1)
            
            print("linear acceleration")
            print("GT: ",gt_linear_acc_w)
            print("GT_error_imu_link",vector_error(gt_linear_acc_w,lin_acc_w_imu_link))
            print("GT_error_base",vector_error(gt_linear_acc_w,lin_acc_w_base))
            print("compare error: ",vector_error(lin_acc_w_base,lin_acc_w_imu_link))
            # print("base: ",lin_acc_w_base)
            # print("imu_link: ",lin_acc_w_imu_link)
            # print("angular acceleration")
            # print("GT error")
            # print("error: ",self.scene.sensors["imu_robot_base"].data.ang_acc_b-self.scene.sensors["imu_robot_imu_link"].data.ang_acc_b)
            # print("base: ",self.scene.sensors["imu_robot_base"].data.ang_acc_b)
            # print("imu_link: ",self.scene.sensors["imu_robot_imu_link"].data.ang_acc_b)

            # skip first step where initial velocity is zero
            if idx < 1000:
                continue
            
            # check the position
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.pos_w,
                self.scene.sensors["imu_robot_imu_link"].data.pos_w,
                rtol=1e-5,
                atol=1e-5,
            )

            # check the orientation
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.quat_w,
                self.scene.sensors["imu_robot_imu_link"].data.quat_w,
                rtol=1e-4,
                atol=1e-4,
            )

            # check the angular velocities of the imus
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.ang_vel_b,
                self.scene.sensors["imu_robot_imu_link"].data.ang_vel_b,
                rtol=1e-4,
                atol=1e-4,
            )

            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.ang_acc_b,
                self.scene.sensors["imu_robot_imu_link"].data.ang_acc_b,
                rtol=1e-4,
                atol=1e-4,
            )

            # calculate direction and magnitude error 
            error_lin_vel_w_imu_link_gt = vector_error(gt_linear_vel_w,lin_vel_w_base)
            error_lin_vel_w_base_gt = vector_error(gt_linear_vel_w,lin_vel_w_imu_link)
            error_lin_vel_w_compare = vector_error(lin_vel_w_base,lin_vel_w_imu_link)

            error_lin_acc_w_imu_link_gt = vector_error(gt_linear_acc_w,lin_acc_w_base)
            error_lin_acc_w_base_gt = vector_error(gt_linear_acc_w,lin_acc_w_imu_link)
            error_lin_acc_w_compare = vector_error(lin_acc_w_base,lin_acc_w_imu_link)


            # print("COM_imu:",self.scene.sensors["imu_robot_imu_link"]._com_pos_b)
            # print("COM_base:",self.scene.sensors["imu_robot_base"]._com_pos_b)
            # check within absolute or relative magnitude error
            # self.assertTrue(torch.max(error_lin_vel_w_imu_link_gt[0]) < 1e-3    or torch.max(error_lin_vel_w_imu_link_gt[1]) < 5e-3)
            # self.assertTrue(torch.max(error_lin_vel_w_base_gt[0]) < 1e-3        or torch.max(error_lin_vel_w_base_gt[1]) < 5e-3)
            self.assertTrue(torch.max(error_lin_vel_w_compare[0]) <  1e-3       or torch.max(error_lin_vel_w_compare[1]) <  5e-3)
            # self.assertTrue(torch.max(error_lin_acc_w_imu_link_gt[0]) <  1e0   or torch.max(error_lin_acc_w_imu_link_gt[1]) < 5e-2)
            # self.assertTrue(torch.max(error_lin_acc_w_base_gt[0]) <  1e0      or torch.max(error_lin_acc_w_base_gt[1]) <  5e-2)
            self.assertTrue(torch.max(error_lin_acc_w_compare[0]) < 1e-3        or torch.max(error_lin_acc_w_compare[1]) <  5e-2)

            # check error dir
            # self.assertTrue(torch.max(error_lin_vel_w_imu_link_gt[2]) < 5e-3)
            # self.assertTrue(torch.max(error_lin_vel_w_base_gt[2]) < 5e-3)
            self.assertTrue(torch.max(error_lin_vel_w_compare[2]) < 5e-3)
            # self.assertTrue(torch.max(error_lin_acc_w_imu_link_gt[2]) < 5e-2)
            # self.assertTrue(torch.max(error_lin_acc_w_base_gt[2]) < 5e-2)
            # self.assertTrue(torch.max(error_lin_acc_w_compare[2]) < 5e-2)


            # torch.testing.assert_close(
            #     self.scene.sensors["imu_robot_base"].data.lin_acc_b,
            #     self.scene.sensors["imu_robot_imu_link"].data.lin_acc_b,
            #     rtol=2e-1,
            #     atol=1e-1,
            # )


            # # check the linear velocity of the imus
            # torch.testing.assert_close(
            #     self.scene.sensors["imu_robot_base"].data.lin_vel_b,
            #     self.scene.sensors["imu_robot_imu_link"].data.lin_vel_b,
            #     rtol=2e-1,
            #     atol=2e-3,
            # )





if __name__ == "__main__":
    run_tests()
