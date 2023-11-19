"""
Semiconductor manufacturing environment for Gymnasium.
"""

from gymnasium.envs.registration import register

from .env import *  # noqa: F403

register(
    id="SemiconductorEnv-v0",
    entry_point="semiconductor_env.env:SemiconductorEnv",
)

__version__ = "0.1.0"
__description__ = "Semiconductor manufacturing environment for Gymnasium"
