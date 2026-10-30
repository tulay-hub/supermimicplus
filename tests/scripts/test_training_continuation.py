"""Resume lineage, additive motion contracts, and expanded evaluation libraries."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unisupermimic.scripts.train_supermimic import prepare_training_run_dir
from unisupermimic.sim2sim import (
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    resolve_sim2sim_config,
    restore_run_motion_library,
    validate_resume_config,
)


def source_run(tmp_path):
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[2] / "src/unisupermimic/conf/supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose("config", overrides=["task=lens110_lafan1_super_mimic/mjwarp"])
    run = tmp_path / "source"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "contract_snapshot": extract_contract_snapshot(cfg),
            }
        )
    )
    checkpoint = run / "last.pt"
    torch.save(
        {
            "iteration": 12000,
            **{key: {} for key in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v")},
        },
        checkpoint,
    )
    return cfg, checkpoint


def extended(cfg, destination):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.env.commands.motion.motion_files.append(
        "src/unisupermimic/assets/motions/lens110/test_extra.csv"
    )
    cfg.env.commands.motion.motion_sha256.append("b" * 64)
    cfg.algo.resume = True
    cfg.algo.iterations = 30000
    cfg.training.log_dir = str(destination)
    cfg.training.resume_to_new_run = True
    cfg.training.allow_motion_library_change = True
    return cfg


def test_fork_preserves_source_and_records_iteration_and_library(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    source_bytes = (checkpoint.parent / "run_config.json").read_bytes()
    checkpoint_bytes = checkpoint.read_bytes()
    destination = tmp_path / "continued"
    cfg = extended(cfg, destination)
    assert (
        prepare_training_run_dir(cfg, checkpoint, {"task": "Lens110Lafan1SuperMimic"})
        == destination
    )
    saved = json.loads((destination / "run_config.json").read_text())
    assert saved["run"]["source_iteration"] == 12000
    assert saved["run"]["source_checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert len(saved["config"]["env"]["commands"]["motion"]["motion_files"]) == 41
    assert (checkpoint.parent / "run_config.json").read_bytes() == source_bytes
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (destination / "resume_lineage.json").is_file()
    with pytest.raises(FileExistsError):
        prepare_training_run_dir(cfg, checkpoint, {})


def test_library_change_requires_explicit_opt_in(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    cfg = extended(cfg, tmp_path / "continued")
    with pytest.raises(CrossBackendIncompatibleError, match="motion_files"):
        validate_resume_config(checkpoint.parent, cfg)


@pytest.mark.parametrize(
    "field,value",
    [("algo.policy_hidden", [512, 256]), ("env.commands.motion.model_sha256", "c" * 64)],
)
def test_library_opt_in_does_not_relax_robot_or_network_contracts(tmp_path, field, value):
    cfg, checkpoint = source_run(tmp_path)
    cfg = extended(cfg, tmp_path / "continued")
    OmegaConf.update(cfg, field, value)
    with pytest.raises(CrossBackendIncompatibleError, match=field):
        validate_resume_config(checkpoint.parent, cfg, allow_motion_library_change=True)


def test_additive_library_must_retain_original_contents(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    cfg = extended(cfg, tmp_path / "continued")
    cfg.env.commands.motion.motion_sha256[0] = "c" * 64
    with pytest.raises(CrossBackendIncompatibleError, match="retain every source motion"):
        validate_resume_config(checkpoint.parent, cfg, allow_motion_library_change=True)


def test_changed_library_cannot_write_into_old_run(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    cfg = extended(cfg, checkpoint.parent)
    with pytest.raises(ValueError, match="differ from its source"):
        prepare_training_run_dir(cfg, checkpoint, {})


def test_evaluation_restores_recorded_expanded_library(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    destination = tmp_path / "continued"
    target = extended(cfg, destination)
    prepare_training_run_dir(target, checkpoint, {})
    restore_run_motion_library(destination, cfg)
    assert len(cfg.env.commands.motion.motion_files) == 41
    assert cfg.env.commands.motion.motion_sha256[-1] == "b" * 64
    resolve_sim2sim_config(destination, cfg, strict=True)


def test_continuation_target_must_advance_iteration(tmp_path):
    cfg, checkpoint = source_run(tmp_path)
    cfg = extended(cfg, tmp_path / "continued")
    cfg.algo.iterations = 12000
    with pytest.raises(ValueError, match="exceed the source iteration"):
        prepare_training_run_dir(cfg, checkpoint, {})
