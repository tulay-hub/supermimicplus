"""Full-21 Lens110 numerical contracts; no training loops or GPU are used."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms import (
    SuperMimicCommand,
    SuperMimicCommandCfg,
    joint_action_target,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP, RewardCfg
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    load_comparison_model,
    scene_for_run,
)
from unisupermimic.tasks.motion_tracking.lens110.supermimic.robot import (
    DEFAULT_Q,
    JOINT_NAMES,
    TRACKED_BODIES,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src/unisupermimic/assets/robots/lens110"
CSV = ROOT / "src/unisupermimic/assets/motions/lens110/dance_supermimic.csv"


@pytest.fixture(scope="module")
def lens():
    if not CSV.is_file():
        pytest.skip("Run tools/prepare_lens110_assets.py before the local Lens110 checks")
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "scene_supermimic.xml"))
    kin = G1Kinematics(
        str(ASSETS / "supermimic.xml"), "cpu", tracked=TRACKED_BODIES, joint_names=JOINT_NAMES
    )
    motion = MotionLib([str(CSV)], kin, "cpu", src_fps=100)
    mdp = MDP(
        kin, motion, torch.tensor(DEFAULT_Q), RewardCfg(**manifest["reward_initial_overrides"])
    )
    return manifest, model, kin, motion, mdp


def test_full21_dimensions_and_model_mass(lens):
    _, model, kin, motion, mdp = lens
    assert (model.nq, model.nv, model.nu) == (28, 27, 21)
    assert model.body_mass.sum() == pytest.approx(22.92451)
    assert kin.num_joints == 21
    assert len(kin.tracked_names) == 14
    assert motion.num_frames == 1797
    frames = torch.tensor([0, 100])
    state = motion.ref_state(frames)
    context = torch.zeros(2, 25)
    context[:, 0] = 1
    assert mdp.obs(state, torch.zeros(2, 21), context, frames).shape == (2, 75)
    assert mdp.goal(frames, 10).shape == (2, 570)
    assert mdp.normalize_context(context).shape == (2, 25)
    assert mdp.critic_features(state, frames, context).shape[-1] == 55 + 21 + 14 * 3 + 3 + 6 + 25
    replay = ReplayBuffer(8, 2, "cpu", num_joints=21, context_dim=25)
    assert replay.data["s"].shape == (8, 2, 55)


def test_fk_and_velocities_match_mujoco_including_fixed_hand_tips(lens):
    _, model, kin, _, _ = lens
    rng = np.random.default_rng(42)
    data = mujoco.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    body_ids = [model.body(name).id for name in TRACKED_BODIES]
    qaddrs = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    daddrs = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    for _ in range(12):
        data.qpos[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[qaddrs] = rng.uniform(kin.q_lo.numpy(), kin.q_hi.numpy())
        data.qvel[:] = rng.uniform(-1, 1, model.nv)
        mujoco.mj_forward(model, data)
        mujoco.mj_jacBody(model, data, jacp, jacr, model.body("pelvis").id)

        def tensor(a):
            return torch.as_tensor(np.array(a), dtype=torch.float64).unsqueeze(0)

        predicted = kin.forward(
            tensor(data.qpos[:3]),
            quat_to_mat(tensor(data.qpos[3:7])),
            tensor(jacp @ data.qvel),
            tensor(jacr @ data.qvel),
            tensor(data.qpos[qaddrs]),
            tensor(data.qvel[daddrs]),
        )
        expected_v, expected_w = [], []
        for body in body_ids:
            mujoco.mj_jacBody(model, data, jacp, jacr, body)
            expected_v.append(jacp @ data.qvel)
            expected_w.append(jacr @ data.qvel)
        for actual, expected in zip(
            predicted,
            (
                data.xpos[body_ids],
                data.xmat[body_ids].reshape(-1, 3, 3),
                np.stack(expected_v),
                np.stack(expected_w),
            ),
        ):
            np.testing.assert_allclose(actual[0].numpy(), expected, atol=4e-6, rtol=4e-6)


def test_all_21_actions_use_default_pose_and_lens110_pd(lens):
    manifest, model, _, _, _ = lens
    default = torch.tensor(DEFAULT_Q).unsqueeze(0)
    action = torch.linspace(-0.2, 0.2, 21).unsqueeze(0)
    context = torch.zeros(1, 25)
    alpha = torch.tensor(manifest["action_scales"])
    target = joint_action_target(default, context, alpha, action)
    torch.testing.assert_close(target, default + alpha * action)
    assert not torch.equal(target[:, 13:], default[:, 13:])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.ctrl[:] = target[0].numpy()
    mujoco.mj_forward(model, data)
    expected_torque = np.asarray(manifest["kp"]) * (data.ctrl - np.asarray(DEFAULT_Q))
    np.testing.assert_allclose(data.actuator_force, expected_torque, atol=2e-6)


def test_reward_root_error_and_action_rate_have_expected_effect(lens):
    _, _, _, motion, mdp = lens
    frames = torch.tensor([0, 200])
    state = motion.ref_state(frames)
    zeros = torch.zeros(2, 21)
    base = mdp.reward(state, frames, zeros, zeros)
    shifted = list(state)
    shifted[0] = shifted[0].clone()
    shifted[0][:, 0] += 0.1
    shifted_reward = mdp.reward(tuple(shifted), frames, zeros, zeros)
    expected_loss = 0.02 * 0.5 * (1 - np.exp(-(0.1**2) / mdp.cfg.root_pos_std**2))
    torch.testing.assert_close(
        base - shifted_reward, torch.full_like(base, expected_loss), atol=1e-6, rtol=1e-5
    )
    varied = zeros.clone()
    varied[:, -1] = 0.2
    penalty = mdp.reward(state, frames, varied, zeros)
    torch.testing.assert_close(
        base - penalty, torch.full_like(base, 0.02 * 0.1 * 0.2**2), atol=1e-7, rtol=1e-3
    )


def test_21_action_policy_and_world_model_support_reward_backprop(lens):
    _, _, _, motion, mdp = lens
    frames = torch.tensor([0, 200])
    state = motion.ref_state(frames)
    context = torch.zeros(2, 25)
    context[:, 0] = 1
    previous = torch.zeros(2, 21)
    obs = mdp.obs(state, previous, context, frames)
    policy = Policy(75, 570, hist=10, num_joints=21, hidden=(32, 16))
    world = WorldModel(hidden=(32, 16), num_joints=21, context_dim=25)
    assert (world.in_dim, world.out_dim) == (101, 54)
    torch.nn.init.normal_(world.net[-1].weight, std=1e-3)
    actions = policy(obs[:, None].expand(-1, 10, -1), mdp.goal(frames, 10))
    assert actions.shape == (2, 21)
    next_state = world.step(state, actions, mdp.normalize_context(context))
    loss = -mdp.reward(next_state, frames + 1, actions, previous).mean()
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert policy.dec[-1].weight.grad[13:].abs().sum() > 0


def test_lens110_reference_comparison_model_has_28_dofs_per_robot(lens):
    model = load_comparison_model(ASSETS / "scene_supermimic.xml")
    assert model.nq == 56


def test_model_content_hash_is_checked_before_environment_construction(lens):
    manifest, _, _, _, _ = lens
    cfg = SuperMimicCommandCfg(
        motion_files=[str(CSV)],
        mjcf_path=str(ASSETS / "supermimic.xml"),
        context_dim=25,
        joint_names=JOINT_NAMES,
        tracked_bodies=TRACKED_BODIES,
        action_scales=manifest["action_scales"],
        model_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="MJCF content hash mismatch"):
        SuperMimicCommand(cfg, SimpleNamespace())


def test_video_uses_recorded_lens110_scene(tmp_path):
    scene = ASSETS / "scene_supermimic.xml"
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {"env": {"scene": {"model_file": str(scene)}}},
            }
        )
    )
    assert scene_for_run(tmp_path) == scene
