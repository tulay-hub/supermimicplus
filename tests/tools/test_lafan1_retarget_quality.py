"""Retiming and geometric quality gates, independently checked against MuJoCo."""

from pathlib import Path

import mujoco
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
    import pytest

    pytest.skip(
        "Run these preprocessing checks with the existing gmr environment", allow_module_level=True
    )
from tools.retarget_lafan1_lens110 import ground, lowest_geometry, retime


def test_local_retiming_preserves_poses_and_caps_all_joint_speeds():
    qpos = np.zeros((5, 28))
    qpos[:, 2] = 0.7
    qpos[:, 3:7] = Rotation.from_euler("z", np.linspace(0, 1, 5)).as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7] = [0, 0.5, 0.51, 0.4, 0.3]
    processed, indices, stretch = retime(qpos, 30, np.ones(21))
    np.testing.assert_allclose(processed[[0, -1]], qpos[[0, -1]], atol=1e-12)
    assert np.max(np.abs(np.diff(processed[:, 7:], axis=0)) * 50) <= 0.9 + 1e-10
    np.testing.assert_allclose(np.linalg.norm(processed[:, 3:7], axis=1), 1)
    assert np.max(stretch) > 1
    assert indices.min() == 0 and indices.max() == 4


def test_analytic_lowest_geometry_matches_mujoco_distance():
    scene = (
        Path(__file__).resolve().parents[2]
        / "src/unisupermimic/assets/robots/lens110/scene_supermimic.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    poses = np.tile(model.key("stand").qpos, (8, 1))
    poses[:, :3] += rng.uniform(-0.1, 0.1, (8, 3))
    poses[:, 3:7] = Rotation.random(8, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    poses[:, 7:] += rng.uniform(-0.1, 0.1, (8, 21))
    actual = lowest_geometry(model, poses)
    expected = []
    floor = model.geom("floor").id
    ids = np.flatnonzero(model.geom_bodyid > 0)
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        expected.append(
            min(mujoco.mj_geomDistance(model, data, int(i), floor, 10, None) for i in ids)
        )
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    fixed, before, delta = ground(model, poses, np.ones(8, dtype=bool))
    assert np.min(lowest_geometry(model, fixed)) >= 0.001 - 1e-9
    np.testing.assert_allclose(fixed[:, 7:], poses[:, 7:])
    np.testing.assert_allclose(before + delta, lowest_geometry(model, fixed), atol=1e-12)
