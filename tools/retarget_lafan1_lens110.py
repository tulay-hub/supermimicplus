#!/usr/bin/env python3
"""G1 LAFAN1 -> Lens110 through the workspace robot_retargeter, with audits.

Run with the existing gmr interpreter. No training or viewer is started.
Raw IK, processed CSVs, source hashes and timing/ground corrections are retained.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing
import os
import pickle
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
RETARGET = WORKSPACE / "tools/retargeting/robot_retargeter"
SOURCE = PROJECT / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"
OUTPUT = PROJECT / "src/unisupermimic/assets/motions/lens110/lafan1_retargeted"
AUDIT = PROJECT / "data/retargeting/lafan1_full21_v1"
SCENE = PROJECT / "src/unisupermimic/assets/robots/lens110/scene_supermimic.xml"
RECIPE = "robot_retargeter_lafan1_lens110_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def specs():
    root = PROJECT / "src/unisupermimic/tasks/motion_tracking"
    return (
        constants("source_g1_spec", root / "g1/supermimic/robot.py"),
        constants("target_lens_spec", root / "lens110/supermimic/robot.py"),
    )


def joint_names(model) -> list[str]:
    return [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == 3]


def speed_limits(names) -> np.ndarray:
    manifest = json.loads((SCENE.parent / "manifest.json").read_text())
    urdf = next(Path(p) for p in manifest["source_files"] if p.endswith(".urdf"))
    if digest(urdf) != manifest["source_files"][str(urdf)]:
        raise ValueError("Canonical training URDF changed")
    limits = {
        j.get("name"): float(j.find("limit").get("velocity"))
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.find("limit") is not None
    }
    values = np.array([limits[n] for n in names])
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Invalid URDF speed limits")
    return values


def retime(qpos: np.ndarray, fps: float, limits: np.ndarray, margin: float = 0.9):
    """Preserve the pose path, increasing each interval enough to satisfy joint speed."""
    if len(qpos) < 2 or not 0 < margin <= 1 or fps <= 0:
        raise ValueError("Invalid trajectory, speed margin or source fps")
    ratios = np.max(np.abs(np.diff(qpos[:, 7:], axis=0)) * fps / limits, axis=1)
    stretch = np.maximum(1.0, ratios / margin)
    timeline = np.r_[0.0, np.cumsum(stretch / fps)]
    samples = np.minimum(np.arange(int(np.ceil(timeline[-1] * 50)) + 1) / 50, timeline[-1])
    result = np.empty((len(samples), qpos.shape[1]))
    for j in [0, 1, 2, *range(7, qpos.shape[1])]:
        result[:, j] = np.interp(samples, timeline, qpos[:, j])
    rotations = Slerp(timeline, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(samples)
    result[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(result)):
        if np.dot(result[i - 1, 3:7], result[i, 3:7]) < 0:
            result[i, 3:7] *= -1
    nearest = np.searchsorted(timeline, samples, side="right") - 1
    return result, nearest.clip(0, len(qpos) - 1), stretch


def lowest_geometry(model, qpos: np.ndarray) -> np.ndarray:
    """Exact support height above z=0 for the model's box/sphere/cylinder geoms."""
    data = mujoco.MjData(model)
    ids = np.flatnonzero(model.geom_bodyid > 0)
    types = model.geom_type[ids]
    size = model.geom_size[ids]
    if not set(types).issubset({2, 3, 4, 5, 6}):
        raise ValueError("Only analytic primitive collision geometry is supported")
    result = np.empty(len(qpos))
    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        rz = data.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, :]
        extent = np.empty(len(ids))
        for kind in set(types):
            mask = types == kind
            if kind == 2:
                extent[mask] = size[mask, 0]
            elif kind == 3:
                extent[mask] = size[mask, 0] + size[mask, 1] * np.abs(rz[mask, 2])
            elif kind == 4:
                extent[mask] = np.linalg.norm(rz[mask] * size[mask], axis=1)
            elif kind == 5:
                z = np.clip(np.abs(rz[mask, 2]), 0, 1)
                extent[mask] = size[mask, 1] * z + size[mask, 0] * np.sqrt(1 - z**2)
            else:
                extent[mask] = np.sum(np.abs(rz[mask]) * size[mask], axis=1)
        result[i] = np.min(data.geom_xpos[ids, 2] - extent)
    return result


def ground(model, qpos: np.ndarray, support: np.ndarray, margin: float = 0.001):
    minimum = lowest_geometry(model, qpos)
    required = margin - minimum
    desired = np.where(support, required, np.maximum(required, 0.0))
    correction = np.maximum(gaussian_filter1d(desired, sigma=2, mode="nearest"), required)
    correction[~support] = np.maximum(correction[~support], 0.0)
    result = qpos.copy()
    result[:, 2] += correction
    return result, minimum, correction


def source_config_model(config: Path) -> Path:
    return (RETARGET / yaml.safe_load(config.read_text())["robot_xml_path"]).resolve()


def convert_one(path: str, max_frames: int = 0) -> dict:
    source = Path(path)
    output = OUTPUT if not max_frames else AUDIT / "probe/validated"
    audit = AUDIT if not max_frames else AUDIT / "probe"
    for directory in (
        output,
        audit / "raw",
        audit / "keypoints",
        audit / "quality",
        audit / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    quality_file = audit / "quality" / f"{source.stem}.json"
    destination = output / source.name
    if quality_file.is_file() and destination.is_file():
        old = json.loads(quality_file.read_text())
        if (
            old.get("recipe") == RECIPE
            and old.get("source_sha256") == digest(source)
            and old.get("output_sha256") == digest(destination)
            and old.get("training_scene_sha256") == digest(SCENE)
            and old.get("accepted")
        ):
            return old
        raise ValueError(f"Existing results differ from this recipe: {source.stem}")
    log = audit / "logs" / f"{source.stem}.log"
    with (
        log.open("w", encoding="utf-8") as stream,
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
    ):
        try:
            started = time.perf_counter()
            os.chdir(RETARGET)
            sys.path.insert(0, str(WORKSPACE / "projects/07_yahmp_lens110/scripts"))
            import mink
            import retarget_yahmp_npz_robot_retargeter as adapter

            source_spec, target_spec = specs()
            src_cfg = RETARGET / "config/robot/g1.yaml"
            dst_cfg = RETARGET / "config/robot/lens110.yaml"
            src_xml, dst_xml = source_config_model(src_cfg), source_config_model(dst_cfg)
            src_model = mujoco.MjModel.from_xml_path(str(src_xml))
            physical = mujoco.MjModel.from_xml_path(str(SCENE))
            if joint_names(src_model) != source_spec.G1_JOINT_NAMES:
                raise ValueError("Source G1 model order differs from LAFAN1 CSV")
            if joint_names(physical) != target_spec.JOINT_NAMES:
                raise ValueError("Training Lens110 model order mismatch")
            raw = np.loadtxt(source, delimiter=",", max_rows=max_frames or None)
            if raw.ndim != 2 or raw.shape[1] != 36 or not np.isfinite(raw).all():
                raise ValueError("Expected finite N x 36 G1 CSV")
            qpos = np.concatenate((raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]), axis=1)
            qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
            keypoints = audit / "keypoints" / f"{source.stem}.pkl"
            adapter.run_source_replay_and_keypoints(
                qpos, 30.0, src_xml, dst_xml, src_cfg, dst_cfg, keypoints
            )
            cfg = yaml.safe_load(dst_cfg.read_text())
            ik = adapter.RobotRetarget(
                model_path=str(dst_xml),
                keypoint_path=str(keypoints),
                ik_match_table=cfg["ik_match_table"],
                solver="daqp",
                verbose=False,
                render_debug=False,
                joints_limit_offset_degrees=cfg.get("joints_limit_offset_degrees", {}),
                contact_body_names=cfg.get("contact_links", []),
                contact_position_cost=float(cfg.get("contact_pos_fixed_factor", 10.0)),
            )
            if joint_names(ik.model) != target_spec.JOINT_NAMES:
                raise ValueError("Retargeter output joint order mismatch")
            lower = np.array([physical.joint(n).range[0] for n in target_spec.JOINT_NAMES])
            upper = np.array([physical.joint(n).range[1] for n in target_spec.JOINT_NAMES])
            for i, name in enumerate(target_spec.JOINT_NAMES):
                j = ik.model.joint(name).id
                ik.model.jnt_range[j] = [
                    max(lower[i], ik.model.jnt_range[j, 0]),
                    min(upper[i], ik.model.jnt_range[j, 1]),
                ]
            ik.ik_limits = [mink.ConfigurationLimit(ik.model)]
            seed = ik.model.qpos0.copy()
            seed[7:] = target_spec.DEFAULT_Q
            seed[:3] = ik.keypoints_pos[0, ik.keypoint_name_to_idx["hips_mean"]]
            seed[3:7] = qpos[0, 3:7]
            result = adapter.retarget_with_warm_start(
                ik, initial_qpos=seed, first_frame_min_iterations=50
            )
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite IK output")
            raw_file = audit / "raw" / source.name
            np.savetxt(
                raw_file,
                np.c_[result[:, :3], result[:, [4, 5, 6, 3]], result[:, 7:]],
                delimiter=",",
                fmt="%.10g",
            )
            excess = max(0.0, float(np.maximum(lower - result[:, 7:], result[:, 7:] - upper).max()))
            if excess > 1e-4:
                raise ValueError(f"IK exceeds physical joint limits: {excess} rad")
            result[:, 7:] = np.clip(result[:, 7:], lower, upper)
            limits = speed_limits(target_spec.JOINT_NAMES)
            retimed, indices, stretch = retime(result, 30.0, limits)
            with keypoints.open("rb") as handle:
                contacts = np.asarray(pickle.load(handle)["contact_states"]).astype(bool)
            support = np.any(contacts, axis=1)[indices]
            processed, minimum, correction = ground(physical, retimed, support)
            speed_ratio = float(np.max(np.abs(np.diff(processed[:, 7:], axis=0)) * 50 / limits))
            min_after = float(np.min(minimum + correction))
            if speed_ratio > 0.9001 or min_after < -1e-5:
                raise ValueError(
                    f"Postprocessing quality gate failed: speed={speed_ratio}, ground={min_after}"
                )
            report = dict(
                recipe=RECIPE,
                name=source.stem,
                source=str(source),
                source_sha256=digest(source),
                source_frames=len(raw),
                source_fps=30,
                source_duration_s=(len(raw) - 1) / 30,
                output=str(destination),
                output_frames=len(processed),
                output_fps=50,
                output_duration_s=(len(processed) - 1) / 50,
                joint_names=target_spec.JOINT_NAMES,
                root_quaternion="xyzw",
                training_scene_sha256=digest(SCENE),
                training_model_sha256=digest(SCENE.with_name("supermimic.xml")),
                source_model_sha256=digest(src_xml),
                ik_model_sha256=digest(dst_xml),
                source_config_sha256=digest(src_cfg),
                target_config_sha256=digest(dst_cfg),
                raw_joint_limit_excess_rad=excess,
                maximum_speed_limit_ratio=speed_ratio,
                raw_minimum_geometry_height_m=float(minimum.min()),
                minimum_geometry_height_m=min_after,
                maximum_interval_time_stretch=float(stretch.max()),
                mean_interval_time_stretch=float(stretch.mean()),
                maximum_root_z_correction_m=float(np.abs(correction).max()),
                maximum_root_z_correction_step_m=float(np.abs(np.diff(correction)).max()),
                first_frame_min_ik_iterations=50,
                accepted=True,
                wall_seconds=time.perf_counter() - started,
                validation_boundary="Numerical/kinematic quality gates; no learned-policy or dynamic-success claim",
            )
            metadata = dict(
                fps=50, joint_names=target_spec.JOINT_NAMES, root_quaternion="xyzw", recipe=RECIPE
            )
            temporary = destination.with_suffix(".csv.tmp")
            np.savetxt(
                temporary,
                np.c_[processed[:, :3], processed[:, [4, 5, 6, 3]], processed[:, 7:]],
                delimiter=",",
                fmt="%.10g",
                header=json.dumps(metadata),
            )
            temporary.replace(destination)
            report["output_sha256"] = digest(destination)
            quality_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            return report
        except Exception as exc:
            traceback.print_exc()
            failure = dict(name=source.stem, accepted=False, error=str(exc), log=str(log))
            quality_file.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n")
            return failure


def write_task(reports: list[dict]) -> None:
    if len(reports) != 40 or not all(r["accepted"] for r in reports):
        raise ValueError("The full-40 task requires 40 accepted conversions")
    reports = sorted(reports, key=lambda r: r["name"])
    root = PROJECT / "src/unisupermimic/conf/supermimic/task/lens110_lafan1_super_mimic"
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(
        defaults=["/task/lens110_super_mimic/base", "_self_"],
        env={
            "commands": {
                "motion": dict(
                    src_fps=50.0,
                    motion_files=[
                        Path(r["output"]).relative_to(PROJECT).as_posix() for r in reports
                    ],
                    motion_sha256=[r["output_sha256"] for r in reports],
                )
            }
        },
        training=dict(task_name="Lens110Lafan1SuperMimic", play_env_num=40),
    )
    (root / "base.yaml").write_text(
        "# @package _global_\n" + yaml.safe_dump(payload, sort_keys=False)
    )
    for backend in ("mjwarp", "mujoco", "motrix"):
        config = dict(
            defaults=["/task/lens110_lafan1_super_mimic/base", "_self_"],
            training=dict(sim_backend=backend),
        )
        if backend == "mjwarp":
            config["training"]["warp_cuda_graphs"] = True
            config["training"]["mjwarp_epa_horizon"] = 128
            config["training"]["mjwarp_nccdmax"] = 64
            config["env"] = {"mjwarp_nconmax": 128, "mjwarp_njmax": 512}
        if backend == "motrix":
            config["env"] = {
                "events": {
                    key: {"params": {"randomize_friction": False}}
                    for key in ("startup_context", "reset_context")
                }
            }
        (root / f"{backend}.yaml").write_text(
            "# @package _global_\n" + yaml.safe_dump(config, sort_keys=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--probe", action="store_true", help="Only 120 source frames of walk1_subject1"
    )
    args = parser.parse_args()
    files = sorted(SOURCE.glob("*.csv"))
    if len(files) != 40:
        raise ValueError(f"Expected 40 source motions, found {len(files)}")
    if args.probe:
        print(
            json.dumps(
                convert_one(str(SOURCE / "walk1_subject1.csv"), 120), indent=2, ensure_ascii=False
            )
        )
        return
    AUDIT.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(convert_one, str(path)): path for path in files}
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            (AUDIT / "progress.json").write_text(
                json.dumps(
                    dict(completed=len(reports), total=40, reports=reports),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"[{len(reports):02d}/40] {result['name']}: "
                + (
                    f"OK, {result['output_frames']} frames, speed={result['maximum_speed_limit_ratio']:.3f}"
                    if result["accepted"]
                    else f"FAILED: {result['error']}"
                ),
                flush=True,
            )
    if not all(r["accepted"] for r in reports):
        raise RuntimeError(
            "Some conversions failed; details are in progress.json and per-clip logs"
        )
    write_task(reports)
    summary = dict(
        recipe=RECIPE,
        clips=40,
        input_frames=sum(r["source_frames"] for r in reports),
        input_hours=sum(r["source_duration_s"] for r in reports) / 3600,
        output_frames=sum(r["output_frames"] for r in reports),
        output_hours=sum(r["output_duration_s"] for r in reports) / 3600,
        maximum_speed_limit_ratio=max(r["maximum_speed_limit_ratio"] for r in reports),
        minimum_geometry_height_m=min(r["minimum_geometry_height_m"] for r in reports),
        wall_seconds=time.time() - started,
        reports=sorted(reports, key=lambda r: r["name"]),
    )
    (AUDIT / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
