"""
Semiconductor manufacturing environment for Gymnasium.
"""

from gymnasium.envs.registration import register

from .env import *

register(
    id="SemiconductorEnv-v0",
    entry_point="semiconductor_env.env:SemiconductorEnv",
    max_episode_steps=1000,
)

__version__ = "0.1.0"
__description__ = "Semiconductor manufacturing environment for Gymnasium"
