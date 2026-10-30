#!/usr/bin/env python3
"""Prepare the full-body Lens110 task from the existing Project 10 assets.

Creates task-local files only. Source model/data are never modified.
The CSV stores root position + xyzw quaternion + 21 named joints at 100 Hz;
the shared MotionLib resamples to 50 Hz and recomputes all velocities.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP, RewardCfg
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.lens110.supermimic.robot import (
    DEFAULT_Q,
    JOINT_NAMES,
    TRACKED_BODIES,
)

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "src/unisupermimic/assets/robots/lens110"
MODEL = ASSETS / "supermimic.xml"
SCENE = ASSETS / "scene_supermimic.xml"
MANIFEST = ASSETS / "manifest.json"
CSV = PROJECT / "src/unisupermimic/assets/motions/lens110/dance_supermimic.csv"
CONFIG = PROJECT / "src/unisupermimic/conf/supermimic/task/lens110_super_mimic"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    bodies = {model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
    return [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]


def geom_distance(model, data, first: int, second: int) -> float:
    return float(mujoco.mj_geomDistance(model, data, first, second, 10.0, None))


def prepare(source: Path) -> None:
    source_model = source / "assets/lens110_dance.xml"
    source_contract = source_model.with_suffix(".json")
    source_motion = source / "data/validated/dance_100hz.npz"
    motion_contract = source_motion.with_suffix(".json")
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    if digest(source_model) != contract["model_sha256"]:
        raise ValueError("Source MJCF differs from Project 10's recorded contract")
    if contract["joint_names"] != JOINT_NAMES:
        raise ValueError("Project 10 joint order differs from the Lens110 contract")
    urdf = Path(contract["urdf"])
    if digest(urdf) != contract["urdf_sha256"]:
        raise ValueError("Source URDF hash differs from Project 10's recorded contract")
    pose_source = urdf.with_name("lens110_dance.py")
    poses = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(ast.parse(pose_source.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "InitialStateCfg"
        for keyword in node.keywords
        if keyword.arg == "joint_pos"
    ]
    if len(poses) != 1 or not np.allclose([poses[0][n] for n in JOINT_NAMES], DEFAULT_Q):
        raise ValueError("Lens110 nominal pose has changed in the canonical training asset")
    motion_meta = json.loads(motion_contract.read_text(encoding="utf-8"))
    if motion_meta["quaternion_order"] != "wxyz":
        raise ValueError("Source NPZ must explicitly declare wxyz root quaternions")
    with np.load(source_motion, allow_pickle=False) as data:
        order = [list(data["joint_names"]).index(n) for n in JOINT_NAMES]
        q = np.asarray(data["joint_pos"][:, order], dtype=np.float64)
        p = np.asarray(data["root_pos"], dtype=np.float64)
        quat = np.asarray(data["root_quat"], dtype=np.float64)
        fps = float(data["fps"])
    if fps != 100.0 or not np.isfinite(np.concatenate((p, quat, q), -1)).all():
        raise ValueError("Expected finite Project 10 motion at 100 Hz")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=2e-5):
        raise ValueError("Source quaternions are not unit length")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(CSV, np.concatenate((p, quat[:, [1, 2, 3, 0]], q), -1), delimiter=",", fmt="%.10g")

    ASSETS.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source_model).getroot()
    root.set("model", "lens110_supermimic_full21_v1")
    world = root.find("worldbody")
    assert world is not None
    floor = copy.deepcopy(world.find("geom[@name='floor']"))
    if floor is None:
        raise ValueError("Source model has no named floor")
    floor.set("friction", "0.3 0.01 0.001")
    for node in list(world):
        if node.tag != "body":
            world.remove(node)
    geom_names = []
    for index, geom in enumerate(world.iter("geom")):
        name = f"lens110_collision_{index:03d}"
        geom.set("name", name)
        geom_names.append(name)
    for side in ("left", "right"):
        elbow = world.find(f".//body[@name='{side}_elbow_link']")
        assert elbow is not None
        tips = [g for g in elbow.findall("geom") if g.get("type", "sphere") == "sphere"]
        if len(tips) != 1:
            raise ValueError(f"Expected the existing {side} hand collision sphere")
        # A massless fixed measurement frame at the existing collision-sphere centre.
        ET.SubElement(elbow, "body", name=f"{side}_hand_tip", pos=tips[0].get("pos", "0 0 0"))
    actuator = root.find("actuator")
    assert actuator is not None
    root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    kp, kd, effort = (
        np.asarray(contract[key], dtype=float) for key in ("kp", "kd", "effort_limit")
    )
    for name, p_gain, d_gain, torque in zip(JOINT_NAMES, kp, kd, effort):
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=str(p_gain),
            kv=str(d_gain),
            ctrllimited="false",
            forcerange=numbers([-torque, torque]),
        )
    option = root.find("option")
    assert option is not None
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    scratch = copy.deepcopy(root)
    scratch.find("worldbody").append(copy.deepcopy(floor))
    scratch_model = mujoco.MjModel.from_xml_string(ET.tostring(scratch, encoding="unicode"))
    scratch_data = mujoco.MjData(scratch_model)
    scratch_data.qpos[:3] = [0, 0, 0.68]
    scratch_data.qpos[7:] = DEFAULT_Q
    mujoco.mj_forward(scratch_model, scratch_data)
    floor_id = scratch_model.geom("floor").id
    minimum = min(
        geom_distance(scratch_model, scratch_data, g, floor_id)
        for g in foot_geom_ids(scratch_model)
    )
    root_height = 0.68 - minimum
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="stand",
        qpos=numbers([0, 0, root_height, 1, 0, 0, 0, *DEFAULT_Q]),
        ctrl=numbers(DEFAULT_Q),
    )
    write_xml(root, MODEL)
    scene = ET.Element("mujoco", model="lens110_supermimic_scene")
    ET.SubElement(scene, "include", file="supermimic.xml")
    scene_world = ET.SubElement(scene, "worldbody")
    scene_world.append(floor)
    ET.SubElement(scene_world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    write_xml(scene, SCENE)

    # Initial geometric scaling only; these tolerances have not been tuned by training.
    body_scale = root_height / 0.76
    reward = dict(
        body_pos_std=0.3 * body_scale,
        root_pos_std=0.3 * body_scale,
        body_linvel_std=body_scale,
        term_height=0.25 * body_scale,
    )
    action_scales = (0.25 * effort / kp).tolist()
    base_path = PROJECT / "src/unisupermimic/conf/supermimic/task/g1_super_mimic/base.yaml"
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    env = cfg["env"]
    env["scene"]["model_file"] = relative(SCENE)
    env["scene"]["entities"]["robot"].update(
        joint_names=JOINT_NAMES,
        actuator_names=JOINT_NAMES,
        body_names=TRACKED_BODIES,
        geom_names=geom_names,
    )
    env["sim_dt"] = 0.002
    command = env["commands"]["motion"]
    command.update(
        mjcf_path=relative(MODEL),
        motion_files=[relative(CSV)],
        src_fps=fps,
        context_dim=25,
        joint_names=JOINT_NAMES,
        tracked_bodies=TRACKED_BODIES,
        action_scales=action_scales,
        model_sha256=digest(MODEL),
        motion_sha256=[digest(CSV)],
        reward_cfg={
            "_target_": "unisupermimic.tasks.motion_tracking.g1.supermimic.mdp.RewardCfg",
            **reward,
        },
    )
    for event in ("startup_context", "reset_context"):
        env["events"][event]["params"]["torso_body_name"] = "torso_yaw_link"
    cfg["training"] = {"play_env_num": 1}
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "base.yaml").write_text(
        "# @package _global_\n" + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    for backend in ("mjwarp", "mujoco", "motrix"):
        backend_cfg = dict(
            defaults=["/task/lens110_super_mimic/base", "_self_"],
            training=dict(task_name="Lens110SuperMimic", sim_backend=backend),
        )
        if backend == "motrix":
            backend_cfg["env"] = {
                "events": {
                    name: {"params": {"randomize_friction": False}}
                    for name in ("startup_context", "reset_context")
                }
            }
        (CONFIG / f"{backend}.yaml").write_text(
            "# @package _global_\n" + yaml.safe_dump(backend_cfg, sort_keys=False), encoding="utf-8"
        )
    manifest = dict(
        contract="lens110_supermimic_full21_v1",
        joint_names=JOINT_NAMES,
        tracked_bodies=TRACKED_BODIES,
        obs_dim=75,
        history=10,
        goal_frame_dim=57,
        goal_frames=10,
        action_dim=21,
        state_dim=55,
        context_dim=25,
        csv_quaternion_order="xyzw",
        internal_quaternion_order="wxyz",
        root_velocity_frame="world",
        control_dt=0.02,
        physics_dt=0.002,
        source_fps=fps,
        source_frames=len(q),
        model_sha256=digest(MODEL),
        scene_sha256=digest(SCENE),
        csv_sha256=digest(CSV),
        kp=kp.tolist(),
        kd=kd.tolist(),
        effort_limit=effort.tolist(),
        action_scales=action_scales,
        default_q=DEFAULT_Q,
        nominal_root_height=root_height,
        geometric_scale=body_scale,
        reward_initial_overrides=reward,
        source_files={
            str(path.resolve()): digest(path)
            for path in (
                source_model,
                source_contract,
                source_motion,
                motion_contract,
                urdf,
                pose_source,
            )
        },
        source_project=str(source.resolve()),
        boundary="Kinematic/data/actuator preparation; no training, convergence or hardware validation",
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, key in ((MODEL, "model_sha256"), (SCENE, "scene_sha256"), (CSV, "csv_sha256")):
        if digest(path) != manifest[key]:
            raise ValueError(f"Prepared asset hash mismatch: {path}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if (model.nq, model.nv, model.nu) != (28, 27, 21):
        raise ValueError("Lens110 model dimensions are incorrect")
    if [model.actuator(i).name for i in range(model.nu)] != JOINT_NAMES:
        raise ValueError("Actuator order differs from the policy contract")
    for i, (kp, kd, effort) in enumerate(
        zip(manifest["kp"], manifest["kd"], manifest["effort_limit"])
    ):
        if not np.isclose(model.actuator_gainprm[i, 0], kp) or not np.isclose(
            -model.actuator_biasprm[i, 2], kd
        ):
            raise ValueError("Implicit PD differs from the recorded Lens110 parameters")
        if not np.allclose(model.actuator_forcerange[i], [-effort, effort]):
            raise ValueError("Actuator torque limits differ from the source")
    kin = G1Kinematics(str(MODEL), "cpu", tracked=TRACKED_BODIES, joint_names=JOINT_NAMES)
    motion = MotionLib([str(CSV)], kin, "cpu", src_fps=manifest["source_fps"])
    mdp = MDP(
        kin, motion, torch.tensor(DEFAULT_Q), RewardCfg(**manifest["reward_initial_overrides"])
    )
    if (mdp.obs_dim, mdp.goal_frame_dim, mdp.num_joints, mdp.context_dim) != (75, 57, 21, 25):
        raise ValueError("Lens110 task dimensions are incorrect")
    violation = torch.maximum(kin.q_lo - motion.q, motion.q - kin.q_hi).max().item()
    if violation > 1e-5:
        raise ValueError(f"Reference violates joint limits by {violation} rad")
    source = Path(manifest["source_project"])
    with np.load(source / "data/validated/dance_100hz.npz", allow_pickle=False) as ref:
        indices = [TRACKED_BODIES.index(str(n)) for n in ref["key_body_names"]]
        error = np.linalg.norm(
            motion.body_p[:, indices].numpy() - ref["body_pos"][::2], axis=-1
        ).max()
    if error > 5e-6:
        raise ValueError(f"Prepared reference FK differs from Project 10: {error} m")
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    feet = foot_geom_ids(model)
    min_distance = 10.0
    for frame in range(motion.num_frames):
        data.qpos[:3] = motion.root_p[frame].numpy()
        # The data stores wxyz; derive it from the interpolated rotation without re-reading CSV.
        data.qpos[3:7] = mat_to_quat(motion.root_R[frame]).numpy()
        data.qpos[7:] = motion.q[frame].numpy()
        mujoco.mj_forward(model, data)
        min_distance = min(min_distance, *(geom_distance(model, data, g, floor) for g in feet))
    if min_distance < -1e-4:
        raise ValueError(f"Prepared feet penetrate the floor: {min_distance} m")
    return dict(
        model_nq=model.nq,
        model_nv=model.nv,
        action_dim=model.nu,
        mass_kg=float(model.body_mass.sum()),
        source_frames=manifest["source_frames"],
        frames_50hz=motion.num_frames,
        duration_s=(motion.num_frames - 1) * 0.02,
        observation_dim=mdp.obs_dim,
        goal_dim=10 * mdp.goal_frame_dim,
        context_dim=mdp.context_dim,
        reference_fk_max_error_m=float(error),
        minimum_static_foot_distance_m=min_distance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=PROJECT.parent / "10_lens110_dance")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        prepare(args.source_project.resolve())
    result = validate()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
