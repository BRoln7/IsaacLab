"""
create some objects, etc, ground, Lights, Xforms, cones and somethings created from URDF, in the scene

usage:
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_objects.py

"""

# launch Isaac sim app first

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="add objects to the scene")
AppLauncher.add_app_launcher_args(parser)
args_client = parser.parse_args()
# construct object 
app_launcher = AppLauncher(args_client)
sim_launcher = app_launcher.app

# reset everything

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    # create ground using Class GroundPlane
    cfg_ground = sim_utils.GroundPlaneCfg()
    # ignore the translation and rotation
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # create light
    cfg_light_distant = sim_utils.DistantLightCfg(intensity=3000.0,color=(0.75, 0.75, 0.75))
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create Xform
    prim_utils.create_prim("/World/objects", "Xform")

    # create cones
    # cfg_cone = sim_utils.ConeCfg(radius=0.15, height=0.5, axis="Z", 
    #                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
    # cfg_cone.func("/World/objects/cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    # cfg_cone.func("/World/objects/cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # create a cone with rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg_cone_rigid.func("/World/objects/cone_rigid", cfg_cone_rigid,
                        translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0))
    cfg_cone_rigid.func("/World/objects/cone1", cfg_cone_rigid, translation=(-1.0, 1.0, 1.0))
    cfg_cone_rigid.func("/World/objects/cone2", cfg_cone_rigid, translation=(-1.0, -1.0, 1.0))    
    
    # create a table from ##.usd
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    # Initialize the simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # set perspective
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    design_scene()

    sim.reset()
    print("scene loads completely")

    while sim_launcher.is_running():
        sim.step()

if __name__=="__main__":
    main()
    sim_launcher.close()
