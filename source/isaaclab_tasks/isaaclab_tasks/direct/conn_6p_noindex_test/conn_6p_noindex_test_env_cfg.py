# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass


@configclass
class Conn6pNoindexTestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 6       # fx, fy, fz, tx, ty, tz
    observation_space = 13  # rel_pos(3) + rel_quat(4) + lin_vel(3) + ang_vel(3)
    state_space = 0

    # viewer
    viewer: ViewerCfg = ViewerCfg(eye=(0.3, 0.3, 0.3), lookat=(0.0, 0.0, 0.0))

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # connectors — USD assets (mass and collision properties defined in USD)
    female_connector_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FemaleConnector",
        spawn=UsdFileCfg(
            usd_path="/workspace/AIITests/USD/ConnectorCADs/FemaleConn_6P_NoIndex.usd",
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(0.0, 0.0, 1.0, 0.0)),
    )
    male_connector_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/MaleConnector",
        spawn=UsdFileCfg(
            usd_path="/workspace/AIITests/USD/ConnectorCADs/MaleConn_6P_NoIndex.usd",
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.070), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # initialization: conical space
    cone_half_angle: float = 30.0           # degrees, half-angle of cone around Z axis
    init_distance_range: tuple = (0.05, 0.15)  # meters [min, max]
    init_rot_x_range: tuple = (-0.1, 0.1)      # rad, X rotation offset range
    init_rot_z_range: tuple = (-0.1, 0.1)      # rad, Z rotation offset range

    # action scales
    force_scale: float = 10.0   # N
    torque_scale: float = 1.0   # N·m

    # reward scales
    rew_scale_distance: float = -1.0
    rew_scale_orientation: float = -0.5
    rew_scale_alive: float = 0.1
    rew_scale_success: float = 10.0

    # termination thresholds
    success_dist_threshold: float = 0.005   # m
    success_orient_threshold: float = 0.05  # rad
    max_distance: float = 0.5              # m, out-of-bounds reset
