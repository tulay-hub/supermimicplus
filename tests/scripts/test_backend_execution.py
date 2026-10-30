"""The explicit physics execution switch stays scoped to the selected backend."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from unisupermimic.scripts.train_supermimic import configure_backend_execution


def test_eager_mjwarp_is_selected_explicitly():
    reasons = []
    env = SimpleNamespace(backend=SimpleNamespace(_disable_cuda_graphs=reasons.append))
    cfg = OmegaConf.create({"training": {"sim_backend": "mjwarp", "warp_cuda_graphs": False}})
    configure_backend_execution(env, cfg)
    assert len(reasons) == 1


@pytest.mark.parametrize("backend,enabled", [("mujoco", False), ("mjwarp", True)])
def test_default_and_other_backends_do_not_change_execution(backend, enabled):
    cfg = OmegaConf.create({"training": {"sim_backend": backend, "warp_cuda_graphs": enabled}})
    configure_backend_execution(object(), cfg)
