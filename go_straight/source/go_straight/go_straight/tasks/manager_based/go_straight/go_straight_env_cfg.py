# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal TurtleBot task: Move straight on X axis."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass


# Observation functions (world frame for consistency with rewards)
def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


# Reward Functions
def reward_forward_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward moving forward in +X direction (world frame)."""
    robot = env.scene["robot"]
    vel_x = robot.data.root_lin_vel_w[:, 0]
    return torch.clamp(vel_x, min=0.0)


def reward_straight_movement(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize sideways movement (Y velocity in world frame)."""
    robot = env.scene["robot"]
    vel_y = robot.data.root_lin_vel_w[:, 1]
    return -torch.abs(vel_y)


def reward_no_spin(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize angular velocity (spinning in world frame)."""
    robot = env.scene["robot"]
    ang_vel_z = robot.data.root_ang_vel_w[:, 2]
    return -torch.abs(ang_vel_z)


def reward_no_spin_quadratic(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize angular velocity quadratically (stronger penalty for larger spins)."""
    robot = env.scene["robot"]
    ang_vel_z = robot.data.root_ang_vel_w[:, 2]
    return -(ang_vel_z ** 2)



''' new change '''
def reward_heading_alignment_gated(env):
    robot = env.scene["robot"]
    vel_x = robot.data.root_lin_vel_w[:, 0]

    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    moving_mask = (vel_x > 0.2).float()   # only care when moving
    return -moving_mask * (yaw ** 2)


# Step counter for logging
_log_step = [0]


def reward_debug_logger(env: ManagerBasedEnv) -> torch.Tensor:
    """Debug: logs wheel velocities and robot state. Returns 0 reward."""
    robot = env.scene["robot"]

    # Joint velocities (wheel_left_joint=index 0, wheel_right_joint=index 1)
    joint_vel = robot.data.joint_vel[0]  # Shape: [num_joints]
    wheel_left_vel = joint_vel[0].item()
    wheel_right_vel = joint_vel[1].item()

    # Robot velocities
    vel_x = robot.data.root_lin_vel_w[0, 0].item()
    vel_y = robot.data.root_lin_vel_w[0, 1].item()
    ang_z = robot.data.root_ang_vel_w[0, 2].item()
    pos_x = robot.data.root_pos_w[0, 0].item()

    _log_step[0] += 1
    # print(f"[Step {_log_step[0]:4d}] wheels: L={wheel_left_vel:6.2f} R={wheel_right_vel:6.2f} | "
    #       f"vel: x={vel_x:5.2f} y={vel_y:5.2f} ang={ang_z:5.2f} | pos_x={pos_x:6.2f}")

    return torch.zeros(env.num_envs, device=env.device)


# Scene Configuration
@configclass
class StraightSceneCfg(InteractiveSceneCfg):
    """Simple scene with ground and robot."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/user/robotics/isaac_sim_projects/turtlebot_maze_rl/assets/robots/turtlebot.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                stiffness=10.0,
                damping=50.0,
                velocity_limit=15.0,
            ),
        },
    )

# MDP Configuration
@configclass
class ActionsCfg:
    """Direct wheel velocity control."""

    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_left_joint", "wheel_right_joint"],
        scale=7.0,
        use_default_offset=False,
        # clip = (-1.0, 1.0),  # tuple not supported, use dict instead
        # clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Minimal observations (world frame for consistency with rewards)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Use world frame velocities to match reward frame
        base_lin_vel = ObsTerm(func=root_lin_vel_w)
        base_ang_vel = ObsTerm(func=root_ang_vel_w)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Simple rewards for going straight."""

    forward = RewTerm(func=reward_forward_velocity, weight=3.0)  # Strong reward for +X
    straight = RewTerm(func=reward_straight_movement, weight=3.0)
    # no_spin = RewTerm(func=reward_no_spin, weight=2.0)  # Strong penalty for spinning
    heading = RewTerm(func=reward_heading_alignment_gated, weight=1.0)
    ''' new change '''
    # ↓ reduce this
    # straight = RewTerm(func=reward_straight_movement, weight=1.0)//////

    # ↓ make angular penalty quadratic (IMPORTANT)
    no_spin = RewTerm(func=reward_no_spin_quadratic, weight=2.0)

    # NEW: heading alignment
    
    
    debug = RewTerm(func=reward_debug_logger, weight=1e-8)  # Tiny weight for logging (won't affect learning)


@configclass
class TerminationsCfg:
    """Only timeout termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Reset robot to origin."""

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


# def override_actions(env: ManagerBasedEnv):
#     robot = env.scene["robot"]

#     # Fixed wheel velocities
#     left = 5.0
#     right = 5.0

#     actions = torch.tensor([[left, right]], device=env.device)
#     robot.set_joint_velocity_target(actions)

# Environment Configuration
@configclass
class TurtlebotStraightEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal config: Learn to move straight on X axis."""

    scene: StraightSceneCfg = StraightSceneCfg(num_envs=32, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4  # 4 * (1/60) 0.066 second per action
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
