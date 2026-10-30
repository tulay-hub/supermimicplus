import pytest
from omegaconf import OmegaConf

from unisupermimic.scripts.train_supermimic import configure_backend_capacity


def test_ccd_budget_preserves_explicit_capacities_and_does_not_stack_wrappers(monkeypatch):
    import mujoco_warp
    from mujoco_warp._src import collision_convex

    monkeypatch.setattr(mujoco_warp, "make_data", lambda *args, **kwargs: kwargs)
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", 24)
    cfg = OmegaConf.create(
        {"training": {"sim_backend": "mjwarp", "mjwarp_epa_horizon": 128, "mjwarp_nccdmax": 64}}
    )
    configure_backend_capacity(cfg)
    assert collision_convex.MJ_MAX_EPAHORIZON == 128
    assert mujoco_warp.make_data(nconmax=128)["nccdmax"] == 64
    assert mujoco_warp.make_data(nccdmax=48)["nccdmax"] == 48
    assert "nccdmax" not in mujoco_warp.make_data(naccdmax=1000)
    cfg.training.mjwarp_nccdmax = 32
    configure_backend_capacity(cfg)
    assert mujoco_warp.make_data(nconmax=128)["nccdmax"] == 32


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_storage_budget_rejects_invalid_values(value):
    cfg = OmegaConf.create({"training": {"sim_backend": "mjwarp", "mjwarp_epa_horizon": value}})
    with pytest.raises(ValueError, match="positive integer"):
        configure_backend_capacity(cfg)


def test_other_backends_do_not_apply_warp_storage_settings():
    configure_backend_capacity(
        OmegaConf.create({"training": {"sim_backend": "mujoco", "mjwarp_epa_horizon": False}})
    )
