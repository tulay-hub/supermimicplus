import pytest
import torch

from unisupermimic.rl.algos.supermimic.physics_capacity import PhysicsCapacityGuard


def guard(flags=0, constraints=190, contacts=100):
    return PhysicsCapacityGuard(
        torch.tensor([flags]),
        torch.tensor([constraints]),
        torch.tensor([contacts]),
        torch.tensor([150]),
        contact_capacity=512,
        constraint_capacity=256,
        ccd_capacity=512,
    )


def test_physics_capacity_records_usage_without_changing_inputs():
    checked = guard()
    checked.check()
    assert checked.snapshot()["peak_constraints_per_world"] == 190
    assert checked.snapshot()["checked_control_steps"] == 1


@pytest.mark.parametrize("kwargs", [{"flags": 1}, {"constraints": 257}, {"contacts": 513}])
def test_physics_capacity_rejects_overflow(kwargs):
    with pytest.raises(RuntimeError, match="capacity overflow"):
        guard(**kwargs).check()


def test_epa_horizon_overflow_is_rejected_and_recorded():
    checked = guard(flags=256)
    with pytest.raises(RuntimeError, match="flags=256"):
        checked.check()
    assert checked.snapshot()["overflow_flags"] == 256
