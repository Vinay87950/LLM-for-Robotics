"""Robosuite environment utilities for evaluation."""

import logging
import robosuite as suite
from robosuite.controllers import load_controller_config

logger = logging.getLogger("robosuite_env")

def build_robosuite_env(
    env_name="Lift",
    robot="Panda",
    camera_resolution=256,
    control_freq=10,
    action_dim=7,
):
    """Create a Robosuite environment for evaluation."""
    ctrl = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_resolution,
        camera_widths=camera_resolution,
        reward_shaping=False,
        control_freq=control_freq,
    )

    env_action_dim = env.action_spec[0].shape[0]
    if env_action_dim != action_dim:
        logger.warning(
            "⚠ action_dim mismatch: env=%d, model=%d",
            env_action_dim, action_dim,
        )

    logger.info(
        "Env: %s | %s | %d×%d | act_dim=%d | ctrl_freq=%d",
        env_name, robot,
        camera_resolution, camera_resolution,
        env_action_dim, control_freq,
    )
    return env
