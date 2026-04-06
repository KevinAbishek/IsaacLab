# conn_6p_noindex_test — Project Context

## Purpose
Connector insertion RL task. A male 6-pin connector (no index key) must be guided to mate with
a fixed female connector using applied forces and torques. Built on the IsaacLab `DirectRLEnv`
framework.

## Directory Structure
```
conn_6p_noindex_test/
├── CONTEXT.md                          # this file
├── __init__.py                         # gym environment registration
├── conn_6p_noindex_test_env.py         # environment logic
├── conn_6p_noindex_test_env_cfg.py     # configuration dataclass
└── agents/
    ├── __init__.py
    └── skrl_ppo_cfg.yaml               # PPO training hyperparameters
```

## Environment Details

### Gym Registration
```
Isaac-Conn-6p-Noindex-Test-Direct-v0
```

### Assets (USD)
```
/workspace/AIITests/USD/ConnectorCADs/FemaleConn_6P_NoIndex.usd   (scale 0.001)
/workspace/AIITests/USD/ConnectorCADs/MaleConn_6P_NoIndex.usd     (scale 0.001)
```

### Initial Poses
| Body | Position (m) | Rotation (wxyz) |
|------|-------------|-----------------|
| Female connector | (0, 0, 0.015) | (0, 0, 1, 0) |
| Male connector | (0, 0, 0.070) | (1, 0, 0, 0) |

Female connector is kinematic (fixed). Male connector is dynamic and receives external wrench.

### Simulation
- `dt = 1/120 s`, `decimation = 2` → policy runs at 60 Hz
- `episode_length_s = 5.0 s`
- `num_envs = 4096`, `env_spacing = 4.0 m`

### Action Space — 6D wrench in male connector body frame
| Index | Meaning | Scale |
|-------|---------|-------|
| 0–2 | Force x, y, z | 10.0 N |
| 3–5 | Torque x, y, z | 1.0 N·m |

Actions are rotated into the world frame before application.

### Observation Space — 13D
| Slice | Content | Frame |
|-------|---------|-------|
| `[0:3]` | Relative position (male − female) | Female connector frame |
| `[3:7]` | Relative quaternion (wxyz) | Female connector frame |
| `[7:10]` | Linear velocity of male | World frame |
| `[10:13]` | Angular velocity of male | World frame |

### Reward Function
```
r = -1.0 * dist
  + -0.5 * orient_err
  +  0.1 * 1  (alive bonus)
  + 10.0 * success
```

### Termination
- **Success**: `dist < 0.005 m` AND `orient_err < 0.05 rad`
- **Out of bounds**: `dist > 0.5 m`
- **Timeout**: episode length exceeded

---

## Current Experiment Goals

1. **Contact force sensors** — Add `ContactSensor` to the male connector to expose net contact
   forces as additional observations (planned obs space → 19D).

2. **Cartesian impedance controller** — Replace raw wrench actions with a spring-damper
   controller that drives the male connector toward a goal pose:
   ```
   F = Kp_pos * pos_err − Kd_pos * lin_vel
   T = Kp_rot * orient_err_vec − Kd_rot * ang_vel
   ```
   RL actions become residual delta corrections on top of the nominal impedance wrench.

3. **Zero-action evaluation** — Run the environment with `actions = zeros(N, 6)` to benchmark
   the impedance controller alone (no RL). This establishes a baseline for subsequent RL training.

---

## Key References
- IsaacLab `ContactSensor`: `source/isaaclab/isaaclab/sensors/contact_sensor/`
- IsaacLab math utils: `source/isaaclab/isaaclab/utils/math.py`
- `DirectRLEnv` base: `source/isaaclab/isaaclab/envs/direct_rl_env.py`
