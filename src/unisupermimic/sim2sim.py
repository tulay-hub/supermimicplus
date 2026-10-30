"""SuperMimic policy-I/O contract snapshots and strict evaluation checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from unilab.utils.sim2sim import DENYLIST as UPSTREAM_DENYLIST
from unilab.utils.sim2sim import (
    WARNING_LIST,
    CrossBackendIncompatibleError,
    policy_load_dim_guard,
)

SUPERMIMIC_DENYLIST = [
    "algo.goal_frames",
    "algo.hist",
    "algo.z_dim",
    "algo.wm_hidden",
    "algo.policy_hidden",
    "algo.critic_hidden",
    "algo.wm_window",
    "algo.horizon",
    "algo.amp",
    "algo.test_hidden_dims",
    "env.commands.motion.goal_frames",
    "env.commands.motion.context_dim",
    "env.commands.motion.motion_files",
    "env.commands.motion.mjcf_path",
    "env.commands.motion.joint_names",
    "env.commands.motion.tracked_bodies",
    "env.commands.motion.action_scales",
    "env.commands.motion.reward_cfg",
    "env.commands.motion.model_sha256",
    "env.commands.motion.motion_sha256",
]
DENYLIST = list(dict.fromkeys([*UPSTREAM_DENYLIST, *SUPERMIMIC_DENYLIST]))
SNAPSHOT_FIELDS = [
    *DENYLIST,
    *WARNING_LIST,
    "training.warp_cuda_graphs",
    "training.mjwarp_epa_horizon",
    "training.mjwarp_nccdmax",
]
ENV_STRUCTURAL_DENYLIST = [path for path in DENYLIST if path.startswith("env.")]


def _unsupported_snapshot(run_dir: Path, reason: str) -> CrossBackendIncompatibleError:
    return CrossBackendIncompatibleError(
        f"Unsupported SuperMimic source run: {reason}\n"
        f"Source run: {run_dir}\n"
        "Expected run_config.json with a canonical contract_snapshot mapping."
    )


def _select(cfg: Any, path: str) -> Any:
    return OmegaConf.select(cfg, path)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def extract_contract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
    cfg: Any = full_cfg if OmegaConf.is_config(full_cfg) else OmegaConf.create(full_cfg)
    snapshot: dict[str, Any] = {}
    for path in SNAPSHOT_FIELDS:
        value = _select(cfg, path)
        if value is not None:
            snapshot[path] = _to_plain(value)
    return snapshot


def _normalize(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        return value.replace(
            "unilab.tasks.motion_tracking.g1.supermimic",
            "unisupermimic.tasks.motion_tracking.g1.supermimic",
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _asset_identity(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    marker = "/assets/"
    if marker in normalized:
        return "assets/" + normalized.split(marker, 1)[1]
    if normalized.startswith("assets/"):
        return normalized
    return normalized


def _normalize_contract_value(path: str, value: Any) -> Any:
    normalized = _normalize(value)
    if path == "env.commands.motion.mjcf_path":
        return _asset_identity(normalized)
    if path == "env.commands.motion.motion_files" and isinstance(normalized, list):
        return [_asset_identity(item) for item in normalized]
    return normalized


def _format_value(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)


def _diff_line(path: str, source_value: Any, target_value: Any) -> str:
    return f"{path}: source={_format_value(source_value)} target={_format_value(target_value)}"


def _asymmetric_line(path: str, value: Any, *, source_present: bool) -> str:
    if source_present:
        return (
            f"{path}: source={_format_value(value)} target=<absent>; "
            "set the target field explicitly"
        )
    return (
        f"{path}: source=<absent> target={_format_value(value)}; "
        "set the source field explicitly in a supported run"
    )


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise _unsupported_snapshot(run_dir, f"{path} does not exist.")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _unsupported_snapshot(run_dir, f"{path} is not valid JSON ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise _unsupported_snapshot(run_dir, f"{path} top level is not a JSON object.")
    snapshot = parsed.get("contract_snapshot")
    if not isinstance(snapshot, dict):
        raise _unsupported_snapshot(run_dir, "contract_snapshot is missing or is not a mapping.")
    return snapshot


MOTION_LIBRARY_FIELDS = (
    "env.commands.motion.motion_files",
    "env.commands.motion.motion_sha256",
)


def restore_run_motion_library(run_dir: str | Path, cfg: DictConfig) -> None:
    """Evaluate the exact recorded library, including additive training stages."""
    snapshot = _read_snapshot(Path(run_dir))
    for path in MOTION_LIBRARY_FIELDS:
        if path in snapshot:
            OmegaConf.update(cfg, path, snapshot[path], merge=False)
    full = json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))
    fps = OmegaConf.select(OmegaConf.create(full.get("config", {})), "env.commands.motion.src_fps")
    if fps is not None:
        OmegaConf.update(cfg, "env.commands.motion.src_fps", fps)


def validate_resume_config(
    run_dir: str | Path, cfg: DictConfig, *, allow_motion_library_change: bool = False
) -> None:
    """Allow an explicit additive library extension while preserving all other contracts."""
    target = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if not isinstance(target, DictConfig):
        raise TypeError("Resume configuration must resolve to a mapping")
    if allow_motion_library_change:
        snapshot = _read_snapshot(Path(run_dir))
        old_files = snapshot.get(MOTION_LIBRARY_FIELDS[0])
        old_hashes = snapshot.get(MOTION_LIBRARY_FIELDS[1])
        new_files = OmegaConf.select(cfg, MOTION_LIBRARY_FIELDS[0])
        new_hashes = OmegaConf.select(cfg, MOTION_LIBRARY_FIELDS[1])
        if not old_files or not old_hashes or new_files is None or new_hashes is None:
            raise ValueError("Library extension requires explicit source and target file hashes")
        if len(old_files) != len(old_hashes) or len(new_files) != len(new_hashes):
            raise ValueError("Motion files and content hashes must have matching lengths")
        identities = [_asset_identity(path) for path in new_files]
        if len(set(identities)) != len(identities):
            raise ValueError("Expanded motion library contains duplicate paths")
        destination = dict(zip(identities, new_hashes, strict=True))
        for path, digest in zip(old_files, old_hashes, strict=True):
            if destination.get(_asset_identity(path)) != digest:
                raise CrossBackendIncompatibleError(
                    f"Additive continuation must retain every source motion unchanged: {path}"
                )
        for path in MOTION_LIBRARY_FIELDS:
            OmegaConf.update(target, path, snapshot[path], merge=False)
    resolve_sim2sim_config(run_dir, target, strict=True)


def resolve_sim2sim_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    algo_name: str | None = None,
    strict: bool = True,
) -> DictConfig | None:
    del algo_name
    if source_run_dir is None:
        return None
    run_dir = Path(source_run_dir)
    snapshot = _read_snapshot(run_dir)
    unknown_paths = sorted(set(snapshot) - set(SNAPSHOT_FIELDS))
    if unknown_paths:
        raise _unsupported_snapshot(
            run_dir,
            "contract_snapshot contains keys outside the SuperMimic schema: "
            + ", ".join(unknown_paths),
        )

    denials: list[str] = []
    for path, source_value in snapshot.items():
        target_value = _select(target_cfg, path)
        if target_value is None:
            if path in ENV_STRUCTURAL_DENYLIST:
                denials.append(_asymmetric_line(path, source_value, source_present=True))
            continue
        if _normalize_contract_value(path, source_value) == _normalize_contract_value(
            path, target_value
        ):
            continue
        line = _diff_line(path, source_value, target_value)
        if path in DENYLIST:
            denials.append(line)
        else:
            print(f"[sim2sim] WARNING override {line}")

    for path in ENV_STRUCTURAL_DENYLIST:
        if path not in snapshot:
            target_value = _select(target_cfg, path)
            if target_value is not None:
                denials.append(_asymmetric_line(path, target_value, source_present=False))

    if denials:
        message = f"SuperMimic sim2sim contract mismatch.\nSource run: {run_dir}\n  " + "\n  ".join(
            denials
        )
        if strict:
            raise CrossBackendIncompatibleError(message)
        print(f"[sim2sim] WARNING (non-strict) {message}")
    return target_cfg


__all__ = [
    "DENYLIST",
    "ENV_STRUCTURAL_DENYLIST",
    "SNAPSHOT_FIELDS",
    "SUPERMIMIC_DENYLIST",
    "extract_contract_snapshot",
    "policy_load_dim_guard",
    "resolve_sim2sim_config",
    "restore_run_motion_library",
    "validate_resume_config",
]
