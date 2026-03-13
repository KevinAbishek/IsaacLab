# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
)

from .conn_6p_noindex_test_env_cfg import Conn6pNoindexTestEnvCfg


class Conn6pNoindexTestEnv(DirectRLEnv):
    cfg: Conn6pNoindexTestEnvCfg

    def __init__(self, cfg: Conn6pNoindexTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # pre-allocate action buffers
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

    def _setup_scene(self):
        self.female_connector = RigidObject(self.cfg.female_connector_cfg)
        self.male_connector = RigidObject(self.cfg.male_connector_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # register in scene
        self.scene.rigid_objects["female_connector"] = self.female_connector
        self.scene.rigid_objects["male_connector"] = self.male_connector
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._forces[:, 0, :] = actions[:, :3] * self.cfg.force_scale
        self._torques[:, 0, :] = actions[:, 3:] * self.cfg.torque_scale

    def _apply_action(self) -> None:
        # actions are in male connector body frame — rotate into world frame before applying
        male_quat = self.male_connector.data.root_link_quat_w  # (N, 4)
        world_forces = quat_apply(male_quat, self._forces[:, 0, :]).unsqueeze(1)   # (N, 1, 3)
        world_torques = quat_apply(male_quat, self._torques[:, 0, :]).unsqueeze(1) # (N, 1, 3)
        self.male_connector.set_external_force_and_torque(
            forces=world_forces, torques=world_torques
        )

    def _get_observations(self) -> dict:
        female_pos = self.female_connector.data.root_link_pos_w    # (N, 3)
        female_quat = self.female_connector.data.root_link_quat_w  # (N, 4) wxyz
        male_pos = self.male_connector.data.root_link_pos_w
        male_quat = self.male_connector.data.root_link_quat_w
        lin_vel = self.male_connector.data.root_com_lin_vel_w      # (N, 3)
        ang_vel = self.male_connector.data.root_com_ang_vel_w      # (N, 3)

        rel_pos = quat_apply_inverse(female_quat, male_pos - female_pos)  # in female frame
        rel_quat = quat_mul(quat_inv(female_quat), male_quat)

        obs = torch.cat([rel_pos, rel_quat, lin_vel, ang_vel], dim=-1)  # (N, 13)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        female_pos = self.female_connector.data.root_link_pos_w
        male_pos = self.male_connector.data.root_link_pos_w
        female_quat = self.female_connector.data.root_link_quat_w
        male_quat = self.male_connector.data.root_link_quat_w

        reward, self._dist, self._orient_err, self._success = compute_rewards(
            female_pos,
            male_pos,
            female_quat,
            male_quat,
            self.cfg.rew_scale_distance,
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_success,
            self.cfg.success_dist_threshold,
            self.cfg.success_orient_threshold,
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_bounds = self._dist > self.cfg.max_distance
        terminated = out_of_bounds | self._success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.male_connector._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # -- reset female connector to default pose (kinematic, stays fixed)
        female_default = self.female_connector.data.default_root_state[env_ids].clone()
        female_default[:, :3] += self.scene.env_origins[env_ids]
        self.female_connector.write_root_link_pose_to_sim(female_default[:, :7], env_ids=env_ids)
        self.female_connector.write_root_com_velocity_to_sim(female_default[:, 7:], env_ids=env_ids)

        # -- sample random starting pose for male connector within cone around Z axis
        half_angle_rad = math.radians(self.cfg.cone_half_angle)

        # azimuth: uniform in [0, 2π)
        phi = sample_uniform(0.0, 2.0 * math.pi, (num_resets,), self.device)
        # elevation: uniform in [0, half_angle_rad]
        theta = sample_uniform(0.0, half_angle_rad, (num_resets,), self.device)
        # distance
        r = sample_uniform(self.cfg.init_distance_range[0], self.cfg.init_distance_range[1], (num_resets,), self.device)

        # Cartesian offset in female local frame (cone around +Z)
        sin_theta = torch.sin(theta)
        offset_x = r * sin_theta * torch.cos(phi)
        offset_y = r * sin_theta * torch.sin(phi)
        offset_z = r * torch.cos(theta)
        cone_offset = torch.stack([offset_x, offset_y, offset_z], dim=-1)  # (num_resets, 3)

        # female position in world frame
        female_pos_w = female_default[:, :3]  # already has env_origins added
        male_pos_w = female_pos_w + cone_offset

        # -- sample orientation offsets around X and Z axes
        rot_x = sample_uniform(
            self.cfg.init_rot_x_range[0], self.cfg.init_rot_x_range[1], (num_resets,), self.device
        )
        rot_z = sample_uniform(
            self.cfg.init_rot_z_range[0], self.cfg.init_rot_z_range[1], (num_resets,), self.device
        )
        zeros = torch.zeros(num_resets, device=self.device)

        # compose orientation: start from female quat, apply X then Z rotation offset
        female_quat_w = female_default[:, 3:7]
        delta_quat_x = quat_from_euler_xyz(rot_x, zeros, zeros)      # roll around X
        delta_quat_z = quat_from_euler_xyz(zeros, zeros, rot_z)       # yaw around Z
        delta_quat = quat_mul(delta_quat_z, delta_quat_x)
        male_quat_w = quat_mul(female_quat_w, delta_quat)

        male_pose_w = torch.cat([male_pos_w, male_quat_w], dim=-1)
        zero_vel = torch.zeros(num_resets, 6, device=self.device)

        self.male_connector.write_root_link_pose_to_sim(male_pose_w, env_ids=env_ids)
        self.male_connector.write_root_com_velocity_to_sim(zero_vel, env_ids=env_ids)


@torch.jit.script
def compute_rewards(
    female_pos: torch.Tensor,
    male_pos: torch.Tensor,
    female_quat: torch.Tensor,
    male_quat: torch.Tensor,
    rew_scale_distance: float,
    rew_scale_orientation: float,
    rew_scale_alive: float,
    rew_scale_success: float,
    success_dist_threshold: float,
    success_orient_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # distance between connectors
    dist = torch.norm(male_pos - female_pos, dim=-1)

    # orientation error relative to identity (aligned with female)
    rel_quat = quat_mul(quat_inv(female_quat), male_quat)
    identity_quat = torch.zeros_like(rel_quat)
    identity_quat[:, 0] = 1.0
    orient_err = quat_error_magnitude(rel_quat, identity_quat)

    success = (dist < success_dist_threshold) & (orient_err < success_orient_threshold)

    reward = (
        rew_scale_distance * dist
        + rew_scale_orientation * orient_err
        + rew_scale_alive * torch.ones_like(dist)
        + rew_scale_success * success.float()
    )
    return reward, dist, orient_err, success
