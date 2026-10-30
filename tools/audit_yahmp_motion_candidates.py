#!/usr/bin/env python3
"""Read-only YAHMP motion screening against the current SuperMimic Lens110 asset.

Passing this audit identifies conversion candidates, not learned-policy success
or dynamic feasibility. Collision/FK checks use sampled reference poses only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent.parent
P07 = WORKSPACE / "projects/07_yahmp_lens110"
PACKAGE = WORKSPACE / "projects/lens110_yahmp_50000_repro_20260923"
ROOT07 = P07 / "data/smoothed_grounded/npz"
ROOT_PACKAGE = PACKAGE / "assets/motions/lens110_smoothed_grounded"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_tags(name):
    """Filename hints only; never assert these are verified motion labels."""
    text = name.lower()
    groups = {
        "walk": ("walk", "walking"),
        "run": ("run", "jog", "sprint"),
        "turn": ("turn", "180", "360", "spin"),
        "jump": ("jump", "hop", "skip"),
        "dance": ("dance", "tangbo", "bangbang", "fast_cars"),
        "martial": ("fight", "kick", "punch", "boxing", "karate"),
        "floor_or_recovery": (
            "roll",
            "getup",
            "get_up",
            "fall",
            "kneel",
            "crouch",
            "squat",
            "sit",
            "lie",
        ),
        "stand": ("stand", "tpose", "t_pose"),
        "sport": ("handball", "basket", "throw", "tennis"),
    }
    tags = [group for group, words in groups.items() if any(word in text for word in words)]
    if name.startswith(("omomo_", "grab_")):
        tags.append("object_interaction_source_review")
    if any(
        word in text
        for word in (
            "box",
            "chair",
            "table",
            "seat",
            "suitcase",
            "ladder",
            "stair",
            "stool",
            "carry",
            "lift",
            "sitdown",
            "sitting",
        )
    ):
        tags.append("external_object_or_support_hint")
    return tags or ["unlabelled"]


def summarize(rows):
    good = [row for row in rows if row.get("offline_candidate")]
    seconds = sum(row.get("duration_s", 0.0) for row in rows)
    return {
        "clips": len(rows),
        "duration_hours": seconds / 3600,
        "offline_candidates": len(good),
        "offline_candidate_hours": sum(row["duration_s"] for row in good) / 3600,
        "reasons": dict(Counter(reason for row in rows for reason in row.get("reasons", []))),
        "sources": dict(Counter(row["source_prefix"] for row in rows)),
        "candidate_sources": dict(Counter(row["source_prefix"] for row in good)),
        "candidate_filename_tags": dict(
            Counter(tag for row in good for tag in row["filename_tags"])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "data/audits/yahmp_motion_candidates_20261029"
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--paths-json", type=Path)
    parser.add_argument("--max-self-penetration", type=float, default=0.01)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("At least two sample poses are required")
    args.output.mkdir(parents=True, exist_ok=True)
    asset = PROJECT / "src/unisupermimic/assets/robots/lens110"
    robot = json.loads((asset / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(asset / "scene_supermimic.xml"))
    data = mujoco.MjData(model)
    names = robot["joint_names"]
    order = json.loads((P07 / "data/isaac_order/conversion_report.json").read_text())
    assert names == order["source_joint_names"], "NPZ grouped joint-order evidence differs"
    assert model.nq == 28 and model.nu == 21
    bounds = np.array([model.joint(name).range for name in names])
    urdf = next(Path(path) for path in robot["source_files"] if path.endswith(".urdf"))
    limits = {
        joint.get("name"): float(joint.find("limit").get("velocity"))
        for joint in ET.parse(urdf).getroot().findall("joint")
        if joint.find("limit") is not None
    }
    speeds = np.array([limits[name] for name in names])
    geometry = np.flatnonzero(model.geom_bodyid > 0)
    floor = model.geom("floor").id
    evidence = {}
    with (P07 / "data/smoothed_grounded/smoothed_manifest.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            evidence[Path(item["output_npz"]).name] = item
    primary = sorted(ROOT_PACKAGE.rglob("*.npz"))
    extras = sorted(
        set((PACKAGE / "delivery/demo_motions").rglob("*.npz"))
        | set((PACKAGE / "examples/custom_motions").rglob("*.npz"))
    )
    if args.paths_json:
        requested = [Path(path) for path in json.loads(args.paths_json.read_text())]
        primary = [path for path in requested if path.is_relative_to(ROOT_PACKAGE)]
        extras = [path for path in requested if not path.is_relative_to(ROOT_PACKAGE)]
    rows = []
    started = time.monotonic()
    output_manifest = args.output / "motions.jsonl"
    with output_manifest.open("w") as output:
        for index, path in enumerate(primary + extras):
            main_set = path.is_relative_to(ROOT_PACKAGE)
            row = {
                "path": str(path),
                "name": path.stem,
                "set": "formal_8288" if main_set else "extra_demo",
                "source_prefix": path.stem.split("_")[0],
                "filename_tags": semantic_tags(path.stem),
                "reasons": [],
                "offline_candidate": False,
                "policy_tracking_evaluated_in_this_audit": False,
            }
            try:
                row["file_sha256"] = digest(path)
                if main_set:
                    relative = path.relative_to(ROOT_PACKAGE)
                    original = ROOT07 / relative
                    row["project07_path"] = str(original)
                    row["package_identical_to_project07"] = (
                        original.is_file() and digest(original) == row["file_sha256"]
                    )
                    old = evidence.get(path.name, {})
                    row["historical_audit_passed"] = old.get("audit", {}).get("passed")
                    row["historical_ground_clearance_m"] = old.get("audit", {}).get(
                        "ground_clearance"
                    )
                    row["overlaps_current_lafan1_recordings"] = (
                        relative.parts[0] == "lafan_resume_20260915"
                    )
                    if row["overlaps_current_lafan1_recordings"]:
                        row["original_recording"] = path.stem.removeprefix("lafan_").rsplit(
                            "_f", 1
                        )[0]
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                for value in arrays.values():
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise ValueError("nonfinite_array")
                fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
                q = np.asarray(arrays["joint_pos"], dtype=np.float64)
                if (
                    not np.isfinite(fps)
                    or fps <= 0
                    or q.ndim != 2
                    or q.shape[1] != 21
                    or len(q) < 2
                ):
                    raise ValueError("fps_or_joint_shape")
                if "joint_names" in arrays:
                    declared = [str(name) for name in arrays["joint_names"]]
                    if len(declared) != 21 or set(declared) != set(names):
                        raise ValueError("declared_joint_names_mismatch")
                    permutation = [declared.index(name) for name in names]
                    q = q[:, permutation]
                    row["joint_order_evidence"] = "embedded_joint_names"
                    row["joint_order_reordered"] = declared != names
                elif main_set:
                    row["joint_order_evidence"] = "project07_export_contract"
                else:
                    row["reasons"].append("missing_joint_order_metadata")
                    row["joint_order_evidence"] = "unverified"
                body_names = [str(name) for name in arrays.get("body_names", [])]
                if "pelvis" not in body_names:
                    row["reasons"].append("missing_pelvis_body_metadata")
                pelvis = body_names.index("pelvis") if "pelvis" in body_names else 0
                p = np.asarray(
                    arrays["root_pos"] if "root_pos" in arrays else arrays["body_pos_w"][:, pelvis],
                    dtype=np.float64,
                )
                quat = np.asarray(
                    arrays["root_quat_w"]
                    if "root_quat_w" in arrays
                    else arrays["body_quat_w"][:, pelvis],
                    dtype=np.float64,
                )
                if p.shape != (len(q), 3) or quat.shape != (len(q), 4):
                    raise ValueError("root_shape")
                norm = np.linalg.norm(quat, axis=1)
                row.update(
                    frames=len(q),
                    fps=fps,
                    duration_s=(len(q) - 1) / fps,
                    quaternion_norm_error=float(np.max(np.abs(norm - 1))),
                )
                if row["quaternion_norm_error"] > 2e-5:
                    row["reasons"].append("quaternion_norm")
                quat = quat / np.maximum(norm[:, None], 1e-12)
                trajectory_hash = hashlib.sha256()
                for value in (
                    np.array([fps], dtype="<f8"),
                    p.astype("<f4"),
                    quat.astype("<f4"),
                    q.astype("<f4"),
                ):
                    trajectory_hash.update(value.tobytes())
                row["trajectory_sha256"] = trajectory_hash.hexdigest()
                order_verified = row["joint_order_evidence"] != "unverified"
                excess = (
                    np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0).max(axis=0)
                    if order_verified
                    else np.zeros(21)
                )
                ratio = (np.abs(np.diff(q, axis=0)) * fps / speeds).max(axis=0)
                row.update(
                    joint_limit_excess_rad=float(excess.max()) if order_verified else None,
                    violating_joints={
                        name: float(value) for name, value in zip(names, excess) if value > 1e-4
                    },
                    max_joint_speed_ratio=float(ratio.max()),
                    overspeed_joints={
                        name: float(value) for name, value in zip(names, ratio) if value > 1.0001
                    },
                    root_speed_max_m_s=float(
                        np.max(np.linalg.norm(np.diff(p, axis=0) * fps, axis=1))
                    ),
                    root_height_span_m=float(np.ptp(p[:, 2])),
                    joint_motion_rms_rad=float(np.sqrt(np.mean(np.ptp(q, axis=0) ** 2))),
                )
                if row["duration_s"] < 0.66:
                    row["reasons"].append("too_short_for_current_imagination_horizon")
                if row["violating_joints"]:
                    row["reasons"].append("current_joint_limits")
                if row["overspeed_joints"]:
                    row["reasons"].append("current_urdf_speed_limits")
                if not row["reasons"]:
                    sampled = (
                        np.arange(len(q))
                        if args.all_frames
                        else np.unique(
                            np.linspace(0, len(q) - 1, min(args.samples, len(q)), dtype=int)
                        )
                    )
                    common = [
                        (i, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
                        for i, name in enumerate(body_names)
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
                    ]
                    minimum, penetration, fk_error = 10.0, 0.0, 0.0
                    for frame in sampled:
                        data.qpos[:3] = p[frame]
                        data.qpos[3:7] = quat[frame]
                        data.qpos[7:] = q[frame]
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                        minimum = min(
                            minimum,
                            *(
                                float(
                                    mujoco.mj_geomDistance(model, data, int(geom), floor, 10, None)
                                )
                                for geom in geometry
                            ),
                        )
                        for contact in data.contact[: data.ncon]:
                            if (
                                model.geom_bodyid[contact.geom1] > 0
                                and model.geom_bodyid[contact.geom2] > 0
                            ):
                                penetration = min(penetration, float(contact.dist))
                        if common:
                            fk_error = max(
                                fk_error,
                                max(
                                    float(
                                        np.max(
                                            np.abs(
                                                data.xpos[body] - arrays["body_pos_w"][frame, old]
                                            )
                                        )
                                    )
                                    for old, body in common
                                ),
                            )
                    row.update(
                        sampled_frames=len(sampled),
                        sampled_min_geometry_height_m=minimum,
                        sampled_self_penetration_m=penetration,
                        sampled_common_body_fk_error_m=fk_error,
                        common_body_count=len(common),
                    )
                    if minimum < -1e-4:
                        row["reasons"].append("current_geometry_ground_penetration")
                    if penetration < -args.max_self_penetration:
                        row["reasons"].append(
                            f"current_enabled_self_collision_over_{args.max_self_penetration * 1000:g}mm"
                        )
                    if fk_error > 1e-4:
                        row["reasons"].append("current_model_fk_mismatch")
                row["offline_candidate"] = not row["reasons"]
            except Exception as error:
                row["reasons"].append(f"read_or_structure_error:{type(error).__name__}:{error}")
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (index + 1) % 1000 == 0:
                output.flush()
                print(
                    json.dumps(
                        {
                            "processed": index + 1,
                            "total": len(primary + extras),
                            "candidates": sum(item["offline_candidate"] for item in rows),
                            "elapsed_s": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    formal = [row for row in rows if row["set"] == "formal_8288"]
    new_sources = [row for row in formal if not row.get("overlaps_current_lafan1_recordings")]
    good = [row for row in rows if row["offline_candidate"]]
    with (args.output / "offline_candidates.csv").open("w", newline="") as output:
        fields = (
            "name",
            "path",
            "set",
            "source_prefix",
            "duration_s",
            "fps",
            "max_joint_speed_ratio",
            "sampled_min_geometry_height_m",
            "sampled_self_penetration_m",
            "joint_motion_rms_rad",
            "filename_tags",
        )
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(good)
    groups = {
        "formal_8288": summarize(formal),
        "new_omomo_amass_sources": summarize(new_sources),
        "lafan_segments": summarize(
            [row for row in formal if row.get("overlaps_current_lafan1_recordings")]
        ),
        "extra_demos": summarize([row for row in rows if row["set"] == "extra_demo"]),
    }
    counts = Counter(row.get("trajectory_sha256") for row in rows if row.get("trajectory_sha256"))
    summary = {
        "model_path": str(asset / "scene_supermimic.xml"),
        "model_sha256": digest(asset / "supermimic.xml"),
        "source_joint_order_evidence": str(P07 / "data/isaac_order/conversion_report.json"),
        "joint_order": names,
        "current_joint_ranges": bounds.tolist(),
        "urdf_speed_limits": speeds.tolist(),
        "samples_per_numerically_eligible_clip": args.samples,
        "geometry_sampling_mode": "all_frames" if args.all_frames else "sampled_frames",
        "max_self_penetration_m": args.max_self_penetration,
        "groups": groups,
        "package_byte_identical_to_project07": sum(
            bool(row.get("package_identical_to_project07")) for row in formal
        ),
        "lafan_original_recordings": sorted(
            {row["original_recording"] for row in formal if row.get("original_recording")}
        ),
        "within_audited_set_exact_duplicate_trajectories": sum(
            value - 1 for value in counts.values()
        ),
        "elapsed_s": time.monotonic() - started,
        "boundary": "Full-file structural/numerical/position/speed screening; sampled current-model FK, ground and enabled collision checks. Not full-trajectory collision certification, dynamic policy evaluation, or hardware validation. Semantic tags are filename hints. Each clip's actual fps must be respected during conversion to 50 Hz.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
