"""Frozen-policy preview of the random held-out reference; preserve the first outcome."""

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import create_env
from unilab.base.process_device import configure_backend_process_device
from unilab.training import ensure_registries

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.physics_capacity import PhysicsCapacityGuard
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import (
    build_env_override,
    configure_backend_capacity,
    configure_backend_execution,
)
from unisupermimic.sim2sim import (
    MOTION_LIBRARY_FIELDS,
    extract_contract_snapshot,
    resolve_sim2sim_config,
    restore_run_motion_library,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import (
    flattened_qpos,
    reference_qpos,
    render_comparison,
)

HERE = Path.cwd() / "outputs/unseen_cmu_91_04"
HERE.mkdir(parents=True, exist_ok=True)
PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference(path, cfg):
    values = np.loadtxt(path, delimiter=",")
    scene = Path(cfg.env.scene.model_file)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    names = list(cfg.env.commands.motion.joint_names)
    bounds = np.array([model.joint(name).range for name in names])
    speed_limits = np.asarray(
        [
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
            10.4,
        ]
    )
    joints = values[:, 7:]
    limit_excess = float(
        np.maximum(np.maximum(bounds[:, 0] - joints, joints - bounds[:, 1]), 0).max()
    )
    speed_ratio = float((np.abs(np.diff(joints, axis=0)) * 50 / speed_limits).max())
    quaternion_error = float(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1).max())
    minimum, penetration = 10.0, 0.0
    floor = model.geom("floor").id
    geometry = np.flatnonzero(model.geom_bodyid > 0)
    for frame in values:
        data.qpos[:3] = frame[:3]
        data.qpos[3:7] = frame[[6, 3, 4, 5]]
        data.qpos[7:] = frame[7:]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        minimum = min(
            minimum,
            *(
                float(mujoco.mj_geomDistance(model, data, int(geom), floor, 10, None))
                for geom in geometry
            ),
        )
        for contact in data.contact[: data.ncon]:
            if model.geom_bodyid[contact.geom1] > 0 and model.geom_bodyid[contact.geom2] > 0:
                penetration = min(penetration, float(contact.dist))
    report = {
        "frames": len(values),
        "fps": 50,
        "duration_seconds": (len(values) - 1) / 50,
        "joint_limit_excess_rad": limit_excess,
        "max_joint_speed_ratio": speed_ratio,
        "quaternion_norm_error": quaternion_error,
        "minimum_geometry_height_m": minimum,
        "maximum_self_penetration_m": penetration,
        "all_values_finite": bool(np.isfinite(values).all()),
    }
    report["accepted"] = bool(
        values.shape[1] == 28
        and report["all_values_finite"]
        and limit_excess <= 1e-4
        and speed_ratio <= 0.9001
        and quaternion_error <= 1e-6
        and minimum >= -1e-4
        and penetration >= -0.001
    )
    (HERE / "reference_check.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise ValueError("Reference quality gate failed before any policy rollout: " + str(report))
    print("REFERENCE_CHECK " + json.dumps(report), flush=True)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    selected = {
        "checkpoint": str(
            PROJECT / "runs" / "lens110_lafan1_yahmp113_full21_30000_20261029" / "ckpt_030000.pt"
        ),
        "random_seed": 3623939731,
        "selected": {
            "path": str(PROJECT / "src/unisupermimic/assets/motions/lens110/heldout/cmu_91_04.csv")
        },
    }
    source = Path(selected["selected"]["path"])
    checkpoint_path = Path(selected["checkpoint"])
    run = checkpoint_path.parent
    cfg_dir = PROJECT / "src/unisupermimic/conf/supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(
            config_name="config",
            overrides=["task=lens110_lafan1_super_mimic/mjwarp", "training.play_only=true"],
        )
    restore_run_motion_library(run, cfg)
    resolve_sim2sim_config(run, cfg, strict=True)
    original_contract = extract_contract_snapshot(cfg)
    reference_file = source
    check_reference(reference_file, cfg)
    cfg.env.commands.motion.motion_files = [str(reference_file)]
    cfg.env.commands.motion.motion_sha256 = [sha(reference_file)]
    cfg.env.commands.motion.src_fps = 50.0
    changed = [
        key
        for key, value in extract_contract_snapshot(cfg).items()
        if original_contract.get(key) != value
    ]
    assert set(changed) <= set(MOTION_LIBRARY_FIELDS), changed
    (HERE / "playback_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=False, indent=2) + "\n"
    )
    ensure_registries()
    configure_backend_process_device("mjwarp", "cuda:0")
    configure_backend_capacity(cfg)
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device="cuda:0"),
    )
    configure_backend_execution(env, cfg)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=env.device, weights_only=True)
        assert checkpoint["iteration"] == 30000
        policy = load_policy(env, checkpoint)
        guard = PhysicsCapacityGuard.from_env(env)
        mdp = env.supermimic_mdp
        motion = mdp.motion
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current = info[SUPERMIMIC_STATE_KEY]
        hist = int(checkpoint["args"]["hist"])
        goals = int(checkpoint["args"]["goal_frames"])
        history = None
        states = [current["s"][0].cpu()]
        frames = [int(current["f"][0])]
        returns = 0.0
        global_errors, relative_errors, root_errors = [], [], []
        hard_termination = False
        expected_steps = int(motion.clip_len[0]) - 1
        for step in range(expected_steps):
            obs = mdp.obs(
                unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"]
            )
            history = (
                obs.unsqueeze(1).repeat(1, hist, 1)
                if history is None
                else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
            )
            action = policy(history, mdp.goal(current["f"], goals))
            assert torch.isfinite(action).all()
            next_frames = torch.minimum(
                current["f"] + 1, motion.clip_end[motion.clip_id[current["f"]]]
            )
            state = env.step(action)
            if guard is not None:
                guard.check()
            transition = state.info[SUPERMIMIC_TRANSITION_KEY]
            assert torch.isfinite(transition["s1"]).all()
            states.append(transition["s1"][0].cpu())
            frames.append(int(next_frames[0]))
            returns += float(transition["r"][0])
            actual = unflatten_state(transition["s1"])
            body_p = mdp.kin.forward(*actual)[0]
            global_errors.append(
                float(torch.linalg.vector_norm(body_p - motion.body_p[next_frames], dim=-1).mean())
            )
            relative_errors.append(
                float(
                    torch.linalg.vector_norm(
                        (body_p - actual[0].unsqueeze(1))
                        - (motion.body_p[next_frames] - motion.root_p[next_frames].unsqueeze(1)),
                        dim=-1,
                    ).mean()
                )
            )
            root_errors.append(
                float(
                    torch.linalg.vector_norm(
                        actual[0][:, :2] - motion.root_p[next_frames, :2], dim=-1
                    )[0]
                )
            )
            if (step + 1) % 250 == 0:
                print(
                    "ROLLOUT " + json.dumps({"steps": step + 1, "expected_steps": expected_steps}),
                    flush=True,
                )
            if bool(transition["done"][0]):
                hard_termination = bool(transition["term"][0])
                break
            current = state.info[SUPERMIMIC_STATE_KEY]
        actual_qpos = flattened_qpos(torch.stack(states)).numpy()
        reference = reference_qpos(motion, torch.tensor(frames, device=env.device)).cpu().numpy()
        np.savez_compressed(
            HERE / "rollout_trajectories.npz",
            policy_qpos=actual_qpos,
            reference_qpos=reference,
            fps=50,
        )
        result = {
            "name": source.stem,
            "source": str(source),
            "source_sha256": sha(source),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha(checkpoint_path),
            "checkpoint_iteration": 30000,
            "backend": "mjwarp",
            "seed": selected["random_seed"],
            "training_reference_count": 153,
            "training_library_modified": False,
            "changed_evaluation_contract_fields": changed,
            "reference_sha256": sha(reference_file),
            "reference_seconds": expected_steps / 50,
            "simulated_seconds": (len(states) - 1) / 50,
            "hard_termination": hard_termination,
            "completed_full_reference": len(states) - 1 == expected_steps and not hard_termination,
            "norm_return": returns / (expected_steps * mdp.max_step_reward),
            "mean_global_body_position_error_m": float(np.mean(global_errors)),
            "mean_root_relative_body_position_error_m": float(np.mean(relative_errors)),
            "maximum_root_xy_error_m": float(np.max(root_errors)),
            "physics_capacity": guard.snapshot() if guard is not None else None,
            "all_policy_actions_and_states_finite": True,
            "global_ghost": True,
            "initialization": "reference first-frame state; no action/observation noise",
            "boundary": "One randomly selected held-out clip, same physics backend; not a general unseen-motion or hardware guarantee.",
        }
        (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()
        env.backend.close()
    render_scene = PROJECT / "src/unisupermimic/assets/robots/lens110/visual/scene_supermimic.xml"
    output = HERE / ("unseen30000_" + source.stem + "_meshes.mp4")
    written = render_comparison(
        reference,
        actual_qpos,
        output,
        render_scene,
        clip_name="UNSEEN | " + source.stem + " | checkpoint 30000",
        fps=25,
        global_ghost=True,
    )
    result.update(video=str(output), video_frames=written, render_scene=str(render_scene))
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("VIDEO " + str(output), flush=True)


if __name__ == "__main__":
    main()
