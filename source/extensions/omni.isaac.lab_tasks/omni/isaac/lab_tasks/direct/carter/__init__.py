# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .carter_env import CarterEnvCfg, CarterEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Carter-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.carter:CarterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CarterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)