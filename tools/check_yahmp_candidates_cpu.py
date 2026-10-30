#!/usr/bin/env python3
"""Evaluate unseen YAHMP references with a frozen SuperMimic policy on CPU.

Writes only audit-local 50 Hz CSVs and reports. This deliberately evaluates a
different reference library; it is not the original training-set benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import create_env
from unilab.base.process_device import configure_backend_process_device
from unilab.training import ensure_registries

import unisupermimic
from unisupermimic.rl.algos.supermimic.geometry import quat_slerp, unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
from unisupermimic.tasks.motion_tracking.lens110.supermimic.robot import JOINT_NAMES

PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_reference(source, output):
    with np.load(source, allow_pickle=False) as data:
        q = np.asarray(data["joint_pos"], dtype=np.float64)
        if "joint_names" in data:
            names = list(data["joint_names"])
            q = q[:, [names.index(name) for name in JOINT_NAMES]]
        fps = float(data["fps"].reshape(-1)[0])
        body = list(data["body_names"]).index("pelvis")
        p = data["root_pos"] if "root_pos" in data else data["body_pos_w"][:, body]
        quat = data["root_quat_w"] if "root_quat_w" in data else data["body_quat_w"][:, body]
    frames = int(np.floor((len(q) - 1) / fps * 50)) + 1
    sample = np.arange(frames, dtype=np.float64) * fps / 50
    lower = np.clip(np.floor(sample).astype(int), 0, len(q) - 2)
    alpha = np.clip(sample - lower, 0, 1)
    position = p[lower] * (1 - alpha[:, None]) + p[lower + 1] * alpha[:, None]
    joints = q[lower] * (1 - alpha[:, None]) + q[lower + 1] * alpha[:, None]
    orientation = quat_slerp(
        torch.as_tensor(quat[lower]), torch.as_tensor(quat[lower + 1]), torch.as_tensor(alpha)
    ).numpy()
    values = np.concatenate((position, orientation[:, [1, 2, 3, 0]], joints), axis=1)
    metadata = {
        "fps": 50,
        "joint_names": JOINT_NAMES,
        "root_quaternion": "xyzw",
        "source_npz": str(source),
        "source_sha256": sha(source),
        "source_fps": fps,
        "purpose": "audit-local unseen-reference CPU evaluation",
    }
    with output.open("w") as stream:
        stream.write("# " + json.dumps(metadata, ensure_ascii=False) + "\n")
        np.savetxt(stream, values, delimiter=",", fmt="%.10g")
    return frames


def convert_training_control(source, output):
    values = np.loadtxt(source, delimiter=",")[:501]
    with output.open("w") as stream:
        stream.write(
            "# "
            + json.dumps(
                {
                    "fps": 50,
                    "joint_names": JOINT_NAMES,
                    "root_quaternion": "xyzw",
                    "source_training_csv": str(source),
                    "source_sha256": sha(source),
                    "purpose": "first 10 seconds of a training reference as a CPU backend control",
                }
            )
            + "\n"
        )
        np.savetxt(stream, values, delimiter=",", fmt="%.10g")
    return len(values)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-json", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-training-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [Path(path) for path in json.loads(args.paths_json.read_text())]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "references_50hz"
    csv_dir.mkdir(exist_ok=True)
    files = [csv_dir / f"{index:02d}_{source.stem}.csv" for index, source in enumerate(sources)]
    lengths = [convert_reference(source, output) for source, output in zip(sources, files)]
    reference_kinds = ["unseen_yahmp_reference"] * len(sources)
    if args.include_training_controls:
        root = PROJECT / "src/unisupermimic/assets/motions/lens110/lafan1_retargeted"
        for name in ("walk1_subject1", "run1_subject2", "dance1_subject1"):
            source = root / f"{name}.csv"
            output = csv_dir / f"{len(files):02d}_control_{name}.csv"
            lengths.append(convert_training_control(source, output))
            files.append(output)
            sources.append(source)
            reference_kinds.append("training_control_first10s")
    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf/supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["task=lens110_lafan1_super_mimic/mujoco"])
    cfg.env.commands.motion.motion_files = [str(path.resolve()) for path in files]
    cfg.env.commands.motion.motion_sha256 = [sha(path) for path in files]
    cfg.env.commands.motion.src_fps = 50.0
    cfg.env.commands.motion.max_episode_steps = max(lengths) + 16
    cfg.env.max_episode_seconds = float((max(lengths) + 16) * 0.02)
    cfg.env.seed = 0
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    metadata = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha(args.ckpt),
        "iteration": checkpoint["iteration"],
        "backend": "mujoco",
        "device": "cpu",
        "sources": [str(source) for source in sources],
        "reference_kinds": reference_kinds,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "boundary": "Zero-shot test on different references and CPU MuJoCo, not the original 40-motion mjwarp benchmark or hardware validation. Failed clips do not prove reference motions dynamically impossible.",
    }
    (args.output / "evaluation_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    ensure_registries()
    configure_backend_process_device("mujoco", "cpu")
    env = create_env(
        cfg,
        num_envs=len(files),
        env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device="cpu"),
    )
    started = time.monotonic()
    try:
        mdp = env.supermimic_mdp
        motion = mdp.motion
        policy = load_policy(env, checkpoint, "cpu")
        hist = int(checkpoint["args"]["hist"])
        goal_frames = int(checkpoint["args"]["goal_frames"])
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current = info[SUPERMIMIC_STATE_KEY]
        returns = torch.zeros(len(files))
        steps = torch.zeros(len(files), dtype=torch.int64)
        finished = torch.zeros(len(files), dtype=torch.bool)
        failed = torch.zeros_like(finished)
        error_sum = torch.zeros(len(files))
        relative_error_sum = torch.zeros(len(files))
        max_root_xy_error = torch.zeros(len(files))
        history = None
        iteration = 0
        while not bool(finished.all()):
            obs = mdp.obs(
                unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"]
            )
            history = (
                obs.unsqueeze(1).repeat(1, hist, 1)
                if history is None
                else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
            )
            goal = mdp.goal(current["f"], goal_frames)
            action = policy(history, goal)
            if not torch.isfinite(action).all():
                raise ValueError("nonfinite policy action")
            next_frames = torch.minimum(
                current["f"] + 1, motion.clip_end[motion.clip_id[current["f"]]]
            )
            state = env.step(action)
            transition = state.info[SUPERMIMIC_TRANSITION_KEY]
            if not torch.isfinite(transition["s1"]).all():
                raise ValueError("nonfinite simulated state")
            actual = unflatten_state(transition["s1"])
            body_p = mdp.kin.forward(*actual)[0]
            error = torch.linalg.vector_norm(body_p - motion.body_p[next_frames], dim=-1).mean(-1)
            relative_error = torch.linalg.vector_norm(
                (body_p - actual[0].unsqueeze(1))
                - (motion.body_p[next_frames] - motion.root_p[next_frames].unsqueeze(1)),
                dim=-1,
            ).mean(-1)
            root_error = torch.linalg.vector_norm(
                actual[0][:, :2] - motion.root_p[next_frames, :2], dim=-1
            )
            alive = ~finished
            returns += torch.where(alive, transition["r"], 0)
            steps += alive
            error_sum += torch.where(alive, error, 0)
            relative_error_sum += torch.where(alive, relative_error, 0)
            max_root_xy_error = torch.maximum(max_root_xy_error, torch.where(alive, root_error, 0))
            failed |= alive & transition["term"]
            finished |= alive & transition["done"]
            current = state.info[SUPERMIMIC_STATE_KEY]
            iteration += 1
            if iteration % 250 == 0:
                print(
                    json.dumps(
                        {
                            "control_steps": iteration,
                            "finished": int(finished.sum()),
                            "failed": int(failed.sum()),
                            "elapsed_s": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if iteration > max(lengths) + 32:
                raise RuntimeError("evaluation did not finish within full reference lengths")
        normalized = returns / ((motion.clip_len - 1).float() * mdp.max_step_reward)
        reports = []
        for index, source in enumerate(sources):
            expected = int(motion.clip_len[index]) - 1
            reports.append(
                {
                    "name": source.stem,
                    "source_reference": str(source),
                    "reference_kind": reference_kinds[index],
                    "full_reference_seconds": expected * 0.02,
                    "evaluated_seconds": int(steps[index]) * 0.02,
                    "completion_ratio": float(steps[index]) / expected,
                    "failed_height_or_tilt": bool(failed[index]),
                    "norm_return": float(normalized[index]),
                    "mean_global_body_position_error_m": float(
                        error_sum[index] / steps[index].clamp(min=1)
                    ),
                    "mean_root_relative_body_position_error_m": float(
                        relative_error_sum[index] / steps[index].clamp(min=1)
                    ),
                    "max_root_xy_error_m": float(max_root_xy_error[index]),
                }
            )
        result = {
            "checkpoint_iteration": checkpoint["iteration"],
            "clips": len(reports),
            "completed_without_hard_termination": sum(
                not report["failed_height_or_tilt"] and report["completion_ratio"] >= 0.999
                for report in reports
            ),
            "norm_return_macro": float(normalized.mean()),
            "elapsed_s": time.monotonic() - started,
            "reports": reports,
            "boundary": metadata["boundary"],
            "all_actions_and_states_finite": True,
            "completion_criterion": {
                "root_height_error_m": mdp.cfg.term_height,
                "root_tilt_matrix_element_error": mdp.cfg.term_tilt,
                "full_reference_required": True,
                "note": "No hard termination is not a global position tracking precision criterion.",
            },
        }
        for kind in ("unseen_yahmp_reference", "training_control_first10s"):
            group = [report for report in reports if report["reference_kind"] == kind]
            result[kind] = {
                "clips": len(group),
                "completed": sum(
                    not report["failed_height_or_tilt"] and report["completion_ratio"] >= 0.999
                    for report in group
                ),
                "mean_norm_return": sum(report["norm_return"] for report in group)
                / max(len(group), 1),
            }
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
