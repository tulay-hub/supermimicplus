"""SuperMimic training and deterministic full-library evaluation entrypoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.base.process_device import (
    apply_collector_tensor_device_override,
    configure_backend_process_device,
)
from unilab.training import ensure_registries, get_log_root, resolve_run_log_dir
from unilab.training.experiment import write_run_config_snapshot
from unilab.training.seed import apply_configured_training_seed
from unilab.training.tensor_runtime import (
    resolve_recorded_collector_tensor_device,
    select_collector_tensor_device,
)
from unilab.utils.checkpoint import get_latest_run, normalize_checkpoint_value
from unilab.utils.device import get_default_device

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.sim2sim import (
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
    restore_run_motion_library,
    validate_resume_config,
)


def build_learner_config(cfg: DictConfig) -> SuperMimicConfig:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    values = {
        field.name: raw[field.name] for field in fields(SuperMimicConfig) if field.name in raw
    }
    for name in ("wm_hidden", "policy_hidden", "critic_hidden", "test_hidden_dims"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    return SuperMimicConfig(**values)


def build_env_override(
    cfg: DictConfig, *, eval_mode: bool = False, learner_device: str | None = None
) -> dict[str, Any]:
    override: dict[str, Any] = EnvConfigAdapter(
        cfg, root_dir=Path.cwd(), algo_name="supermimic"
    ).build_task_env_cfg_override()
    override = apply_collector_tensor_device_override(
        override,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=learner_device,
    )
    if not eval_mode:
        return override
    commands = dict(override["commands"])
    motion = dict(commands["motion"])
    motion["eval_mode"] = True
    motion["reset_noise"] = False
    commands["motion"] = motion
    override["commands"] = commands
    override["auto_reset"] = True
    return override


def configure_backend_capacity(cfg: DictConfig) -> None:
    """Apply task-owned CCD storage limits before backend/graph construction.

    UniSim does not yet expose nccdmax, and MuJoCo Warp fixes EPA horizon
    storage to a module constant. Keep this compatibility adapter in this
    process rather than changing the installed dependency or robot geometry.
    """
    if str(cfg.training.sim_backend) != "mjwarp":
        return
    horizon = OmegaConf.select(cfg, "training.mjwarp_epa_horizon")
    ccd = OmegaConf.select(cfg, "training.mjwarp_nccdmax")
    for name, value in (("mjwarp_epa_horizon", horizon), ("mjwarp_nccdmax", ccd)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"training.{name} must be a positive integer or null")
    if horizon is None and ccd is None:
        return

    import mujoco_warp
    from mujoco_warp._src import collision_convex

    if horizon is not None:
        collision_convex.MJ_MAX_EPAHORIZON = horizon
    if ccd is not None:
        original = getattr(mujoco_warp.make_data, "_unisupermimic_original", mujoco_warp.make_data)

        def make_data(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("nccdmax") is None and kwargs.get("naccdmax") is None:
                kwargs["nccdmax"] = ccd
            return original(*args, **kwargs)

        setattr(make_data, "_unisupermimic_original", original)
        mujoco_warp.make_data = make_data
    print(f"[runtime] mjwarp storage: EPA horizon={horizon}, CCD slots/world={ccd}")


def configure_backend_execution(env: Any, cfg: DictConfig) -> None:
    """Select the existing eager physics path without changing task mathematics."""
    enabled = OmegaConf.select(cfg, "training.warp_cuda_graphs", default=True)
    if not isinstance(enabled, bool):
        raise TypeError("training.warp_cuda_graphs must be a boolean")
    if str(cfg.training.sim_backend) != "mjwarp" or enabled:
        return
    disable = getattr(env.backend, "_disable_cuda_graphs", None)
    if not callable(disable):
        raise RuntimeError("Installed mjwarp backend cannot select eager execution")
    disable("Explicit training.warp_cuda_graphs=false to reduce physics graph memory")
    print("[runtime] mjwarp uses eager physics stepping; CUDA graphs disabled")


def resolve_supermimic_checkpoint(cfg: DictConfig) -> Path:
    root_dir = Path.cwd()
    task_log_root: Path = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    selected_run = str(cfg.algo.load_run)
    run_dir: Path | None
    if selected_run == "-1":
        run_dir = get_latest_run(task_log_root)
    else:
        candidate = Path(selected_run).expanduser()
        if not candidate.exists():
            candidate = task_log_root / selected_run
        run_dir = candidate
    if run_dir is None:
        raise FileNotFoundError(f"No SuperMimic run found under {task_log_root}")
    if run_dir.is_file():
        return run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"SuperMimic run does not exist: {run_dir}")
    checkpoint: str | None = normalize_checkpoint_value(
        OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    )
    if checkpoint is None:
        path = run_dir / "last.pt"
    elif checkpoint.isdigit():
        path = run_dir / f"ckpt_{int(checkpoint):06d}.pt"
    else:
        path = run_dir / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"SuperMimic checkpoint does not exist: {path}")
    return path


def _motion_count(cfg: DictConfig) -> int:
    files = OmegaConf.select(cfg, "env.commands.motion.motion_files")
    if files is None or len(files) == 0:
        raise ValueError("SuperMimic evaluation requires a nonempty motion library")
    return len(files)


def _close_env(env: Any) -> None:
    env.close()
    env.backend.close()


def prepare_training_run_dir(
    cfg: DictConfig, checkpoint: Path | None, run_metadata: dict[str, Any]
) -> Path:
    fork = OmegaConf.select(cfg, "training.resume_to_new_run", default=False)
    change = OmegaConf.select(cfg, "training.allow_motion_library_change", default=False)
    if not isinstance(fork, bool) or not isinstance(change, bool):
        raise TypeError("Continuation flags must be booleans")
    if checkpoint is None and (fork or change):
        raise ValueError("Continuation flags require algo.resume=true and a checkpoint")
    if change and not fork:
        raise ValueError("A changed motion library requires a new run directory")
    if checkpoint is not None:
        validate_resume_config(checkpoint.parent, cfg, allow_motion_library_change=change)
        requested = OmegaConf.select(cfg, "training.log_dir")
        if not fork:
            if (
                requested is not None
                and Path(str(requested)).expanduser().resolve() != checkpoint.parent.resolve()
            ):
                raise ValueError(
                    "Set training.resume_to_new_run=true to resume into another directory"
                )
            return checkpoint.parent
        if requested is None:
            raise ValueError("A new continuation run requires explicit training.log_dir")
        log_dir = Path(str(requested)).expanduser()
        if log_dir.resolve() == checkpoint.parent.resolve():
            raise ValueError("A new continuation directory must differ from its source run")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required = {"iteration", "wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v"}
        if required - saved.keys():
            raise ValueError("Continuation requires a full last.pt with critic and optimizers")
        source_iteration = int(saved["iteration"])
        if int(cfg.algo.iterations) <= source_iteration:
            raise ValueError("Continuation target must exceed the source iteration")
        del saved
        run_metadata = dict(
            run_metadata,
            source_checkpoint=str(checkpoint.resolve()),
            source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source_iteration=source_iteration,
            additive_motion_library=change,
        )
    else:
        log_dir = Path(resolve_run_log_dir(cfg))
    if (
        (log_dir / "run_config.json").exists()
        or (log_dir / "train_log.csv").exists()
        or any(log_dir.glob("*.pt"))
    ):
        raise FileExistsError(f"Refusing to overwrite an existing run: {log_dir}")
    write_run_config_snapshot(
        log_dir,
        run_metadata=run_metadata,
        full_cfg=cfg,
        contract_snapshot=extract_contract_snapshot(cfg),
    )
    if checkpoint is not None:
        (log_dir / "resume_lineage.json").write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return log_dir


def main(cfg: DictConfig) -> None:
    ensure_registries()
    if float(cfg.algo.gamma) != float(cfg.env.commands.motion.gamma):
        raise ValueError("SuperMimic requires algo.gamma == env.commands.motion.gamma")
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    play_only = bool(cfg.training.play_only)
    checkpoint = resolve_supermimic_checkpoint(cfg) if play_only or cfg.algo.resume else None
    if play_only:
        assert checkpoint is not None
        restore_run_motion_library(checkpoint.parent, cfg)
    env_override = build_env_override(cfg, eval_mode=play_only, learner_device=str(device))
    if play_only:
        assert checkpoint is not None
        resolve_sim2sim_config(
            checkpoint.parent,
            cfg,
            algo_name="supermimic",
            strict=bool(cfg.training.sim2sim_strict),
        )
        log_dir = checkpoint.parent
    else:
        log_dir = prepare_training_run_dir(
            cfg,
            checkpoint,
            {
                "algo": "supermimic",
                "task": str(cfg.training.task_name),
                "sim_backend": str(cfg.training.sim_backend),
                "device": str(device),
                "collector_tensor_device": resolve_recorded_collector_tensor_device(
                    str(cfg.training.sim_backend), env_override
                ),
                "seed": seed_info.to_dict(),
            },
        )

    env = None
    try:
        configure_backend_capacity(cfg)
        env = create_env(
            cfg,
            num_envs=_motion_count(cfg) if play_only else int(cfg.algo.num_envs),
            env_cfg_override=env_override,
        )
        configure_backend_execution(env, cfg)
        if play_only:
            from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import evaluate

            assert checkpoint is not None
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=env.supermimic_mdp.obs_dim,
                env_action_dim=env.action_manager.total_action_dim,
                algo_name="supermimic",
            ):
                result = evaluate(env, saved)
            destination = checkpoint.with_name(checkpoint.stem + "_eval.json")
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
            print(json.dumps(result, indent=2))
            return

        learner = SuperMimicLearner(
            env,
            build_learner_config(cfg),
            log_dir,
            device=str(device),
            backend=str(cfg.training.sim_backend),
        )
        summary = learner.train(resume=checkpoint)
        print(json.dumps(summary, indent=2))
    finally:
        if env is not None:
            _close_env(env)


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
