#!/usr/bin/env python3
"""Independently re-read all processed CSVs and check C++ MuJoCo ground distances."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/retargeting/lafan1_full21_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((AUDIT / "manifest.json").read_text())
    asset = ROOT / "src/unisupermimic/assets/robots/lens110"
    robot = json.loads((asset / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(asset / "scene_supermimic.xml"))
    data = mujoco.MjData(model)
    names = robot["joint_names"]
    bounds = np.array([model.joint(name).range for name in names])
    urdf = next(Path(p) for p in robot["source_files"] if p.endswith(".urdf"))
    limit_map = {
        j.get("name"): float(j.find("limit").get("velocity"))
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.find("limit") is not None
    }
    speeds = np.array([limit_map[n] for n in names])
    floor = model.geom("floor").id
    geometry = np.flatnonzero(model.geom_bodyid > 0)
    sources = {
        p.stem
        for p in (ROOT / "src/unisupermimic/assets/motions/g1/lafan1_supermimic").glob("*.csv")
    }
    if len(manifest["reports"]) != 40 or sources != {r["name"] for r in manifest["reports"]}:
        raise ValueError("Input/output motion set mismatch")
    rows = []
    for report in manifest["reports"]:
        if (
            sha(report["source"]) != report["source_sha256"]
            or sha(report["output"]) != report["output_sha256"]
        ):
            raise ValueError(f"Content hash mismatch: {report['name']}")
        path = Path(report["output"])
        with path.open() as stream:
            metadata = json.loads(stream.readline()[2:])
        if (
            metadata["joint_names"] != names
            or metadata["root_quaternion"] != "xyzw"
            or metadata["fps"] != 50
        ):
            raise ValueError("CSV metadata mismatch")
        values = np.loadtxt(path, delimiter=",")
        if values.shape != (report["output_frames"], 28) or not np.isfinite(values).all():
            raise ValueError("CSV shape or finite check failed")
        q = values[:, 7:]
        excess = max(0.0, float(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]).max()))
        ratio = float(np.max(np.abs(np.diff(q, axis=0)) * 50 / speeds))
        quaternion_error = float(np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)))
        minimum = 10.0
        sampled = np.unique(np.linspace(0, len(values) - 1, min(128, len(values)), dtype=int))
        for i in sampled:
            data.qpos[:3] = values[i, :3]
            data.qpos[3:7] = values[i, [6, 3, 4, 5]]
            data.qpos[7:] = q[i]
            mujoco.mj_kinematics(model, data)
            minimum = min(
                minimum,
                *(
                    float(mujoco.mj_geomDistance(model, data, int(g), floor, 10, None))
                    for g in geometry
                ),
            )
        if excess > 1e-5 or ratio > 0.9001 or quaternion_error > 1e-6 or minimum < -1e-4:
            raise ValueError(
                f"Independent check failed for {report['name']}: {excess}, {ratio}, {quaternion_error}, {minimum}"
            )
        rows.append(
            dict(
                name=report["name"],
                frames=len(values),
                speed_limit_ratio=ratio,
                joint_limit_excess_rad=excess,
                quaternion_norm_error=quaternion_error,
                mujoco_sampled_minimum_distance_m=minimum,
                sampled_frames=len(sampled),
            )
        )
    result = dict(
        clips=len(rows),
        frames=sum(r["frames"] for r in rows),
        max_speed_limit_ratio=max(r["speed_limit_ratio"] for r in rows),
        max_joint_limit_excess_rad=max(r["joint_limit_excess_rad"] for r in rows),
        cpp_sampled_min_distance_m=min(r["mujoco_sampled_minimum_distance_m"] for r in rows),
        source_hashes_unchanged=True,
        all_passed=True,
        reports=rows,
        boundary="Numerical and sampled collision geometry validation; not dynamic tracking evaluation",
    )
    (AUDIT / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
