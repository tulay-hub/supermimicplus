"""Register Lens110 using the shared, robot-configurable SuperMimic environment."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from unisupermimic.tasks.motion_tracking.g1.supermimic.env import (
    SUPPORTED_BACKENDS,
    make_supermimic_env,
)

TASK_NAME = "Lens110SuperMimic"
for task_name in (TASK_NAME, "Lens110Lafan1SuperMimic"):
    registry.register_env_config(task_name, ManagerBasedRlEnvCfg)
    for backend in SUPPORTED_BACKENDS:
        registry.register_env(task_name, make_supermimic_env, sim_backend=backend)
