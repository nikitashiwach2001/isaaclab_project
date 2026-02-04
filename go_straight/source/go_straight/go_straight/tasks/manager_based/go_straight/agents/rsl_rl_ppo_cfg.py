# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TurtlebotMazePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128  # CRITICAL: Much higher for 1 env
    max_iterations = 600
    save_interval = 100
    log_interval = 1  # Log every iteration to see detailed reward breakdown

    experiment_name = "turtlebot_straight"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, #0.001,
        num_learning_epochs=4,
        num_mini_batches=4,  # Reduced from 4
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl= 0.01, #0.01,
        max_grad_norm=0.5,
    )