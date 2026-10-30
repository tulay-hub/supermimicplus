#!/usr/bin/env python3
"""Convert the audited 113 Lens110 clips and emit an additive 153-clip experiment."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from check_yahmp_candidates_cpu import convert_reference, sha

PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "data/audits/yahmp_motion_candidates_20261029"
OUTPUT = PROJECT / "src/unisupermimic/assets/motions/lens110/yahmp_expansion113_v1"
PROVENANCE = PROJECT / "data/retargeting/yahmp_expansion113_v1"
SOURCE_RUN = PROJECT / "runs/lens110_lafan1_full21_20261028"
EXPERIMENT = (
    PROJECT / "src/unisupermimic/conf/supermimic/experiment/lens110_lafan1_yahmp113_30000.yaml"
)


def publish(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to replace different existing contents: {path}")
    if not path.exists():
        path.write_text(text)


def main():
    selected = json.loads((AUDIT / "candidate_manifest.json").read_text())
    assert selected["recommended_clips"] == 113
    cfg = json.loads((SOURCE_RUN / "run_config.json").read_text())["config"]
    motion_cfg = cfg["env"]["commands"]["motion"]
    old_files = motion_cfg["motion_files"]
    old_hashes = motion_cfg["motion_sha256"]
    assert len(old_files) == len(old_hashes) == 40
    for path, expected in zip(old_files, old_hashes):
        assert sha(PROJECT / path) == expected, path
    asset = PROJECT / "src/unisupermimic/assets/robots/lens110"
    assert sha(asset / "supermimic.xml") == selected["model_sha256"]
    model = mujoco.MjModel.from_xml_path(str(asset / "scene_supermimic.xml"))
    data = mujoco.MjData(model)
    names = motion_cfg["joint_names"]
    bounds = np.array([model.joint(name).range for name in names])
    # The immutable candidate manifest records the URDF limits used in screening.
    speeds = np.asarray(json.loads((AUDIT / "summary.json").read_text())["urdf_speed_limits"])
    floor = model.geom("floor").id
    geometry = np.flatnonzero(model.geom_bodyid > 0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, item in enumerate(selected["motions"]):
        source = Path(item["path"])
        if sha(source) != item["file_sha256"]:
            raise ValueError(f"Candidate source changed: {source}")
        path = OUTPUT / f"{source.stem}.csv"
        temporary = path.with_suffix(".csv.preparing")
        convert_reference(source, temporary)
        with temporary.open() as stream:
            metadata = json.loads(stream.readline()[2:])
            body = stream.read()
        metadata.update(
            purpose="additive 113-clip training expansion to iteration 30000",
            model_sha256=selected["model_sha256"],
        )
        temporary.write_text("# " + json.dumps(metadata, ensure_ascii=False) + "\n" + body)
        if path.exists():
            if sha(path) != sha(temporary):
                raise FileExistsError(f"Existing derivative differs: {path}")
            temporary.unlink()
        else:
            temporary.replace(path)
        values = np.loadtxt(path, delimiter=",")
        assert values.shape[1] == 28 and np.isfinite(values).all()
        q = values[:, 7:]
        excess = float(np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0).max())
        ratio = float((np.abs(np.diff(q, axis=0)) * 50 / speeds).max())
        quaternion_error = float(np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)))
        minimum, self_penetration = 10.0, 0.0
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
                    self_penetration = min(self_penetration, float(contact.dist))
        accepted = (
            excess <= 1e-4
            and ratio <= 0.9001
            and quaternion_error <= 1e-6
            and minimum >= -1e-4
            and self_penetration >= -0.001
        )
        reports.append(
            {
                "name": source.stem,
                "source": str(source),
                "source_sha256": item["file_sha256"],
                "output": str(path),
                "output_sha256": sha(path),
                "source_fps": item["fps"],
                "output_fps": 50,
                "frames": len(values),
                "duration_s": (len(values) - 1) / 50,
                "joint_limit_excess_rad": excess,
                "max_joint_speed_ratio": ratio,
                "quaternion_norm_error": quaternion_error,
                "all_frame_min_geometry_height_m": minimum,
                "all_frame_self_penetration_m": self_penetration,
                "accepted": accepted,
            }
        )
        if (index + 1) % 20 == 0:
            print(f"checked {index + 1}/113", flush=True)
    new_files = [str(Path(row["output"]).relative_to(PROJECT)) for row in reports]
    files = old_files + new_files
    assert len(files) == len(set(Path(path).stem for path in files)) == 153
    hashes = old_hashes + [row["output_sha256"] for row in reports]
    summary = {
        "source_run": str(SOURCE_RUN),
        "source_candidate_manifest": str(AUDIT / "candidate_manifest.json"),
        "source_candidate_manifest_sha256": sha(AUDIT / "candidate_manifest.json"),
        "model_sha256": selected["model_sha256"],
        "original_clips": 40,
        "added_clips": 113,
        "total_clips": 153,
        "added_frames": sum(row["frames"] for row in reports),
        "added_hours": sum(row["duration_s"] for row in reports) / 3600,
        "all_accepted": all(row["accepted"] for row in reports),
        "reports": reports,
        "boundary": "Every resampled reference frame checked numerically and against current enabled collision geometry; not independent policy tracking or hardware validation.",
    }
    (PROVENANCE / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if not summary["all_accepted"]:
        raise ValueError(
            "Resampled quality check failed: "
            + ", ".join(row["name"] for row in reports if not row["accepted"])
        )
    experiment = {
        "env": {
            "commands": {
                "motion": {"src_fps": 50.0, "motion_files": files, "motion_sha256": hashes}
            }
        },
        "algo": {"iterations": 30000},
        "training": {"resume_to_new_run": True, "allow_motion_library_change": True},
    }
    publish(
        EXPERIMENT,
        "# @package _global_\n" + yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True),
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
