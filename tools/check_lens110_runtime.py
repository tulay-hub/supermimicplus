#!/usr/bin/env python3
"""Bounded Lens110 environment check: no optimizer, learner, or training loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from unilab.base.config_adapter import create_env
from unilab.base.process_device import configure_backend_process_device
from unilab.training import ensure_registries

import unisupermimic
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.physics_capacity import PhysicsCapacityGuard
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import (
    build_env_override,
    configure_backend_capacity,
    configure_backend_execution,
)


def check(
    backend: str, steps: int = 8, task: str = "lens110_super_mimic", all_clips: bool = False
) -> dict:
    if not 1 <= steps <= 100:
        raise ValueError("A runtime check must have 1 to 100 steps")
    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf/supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[f"task={task}/{backend}"])
    ensure_registries()
    tensor_device = "cuda:0" if backend == "mjwarp" else "cpu"
    configure_backend_process_device(backend, tensor_device)
    count = len(cfg.env.commands.motion.motion_files) if all_clips else 2
    configure_backend_capacity(cfg)
    env = create_env(
        cfg,
        num_envs=count,
        env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device=tensor_device),
    )
    configure_backend_execution(env, cfg)
    try:
        capacity = PhysicsCapacityGuard.from_env(env) if backend == "mjwarp" else None
        mdp = env.supermimic_mdp
        frames = (
            mdp.motion.clip_start.clone()
            if all_clips
            else torch.tensor([0, 100], device=env.device)
        )
        observations, info = env.reset_to_frames(frames)
        current = info[SUPERMIMIC_STATE_KEY]
        actual = unflatten_state(current["s"])
        expected = mdp.ref_state(frames)
        reset_error = max((a - b).abs().max().item() for a, b in zip(actual, expected))
        if reset_error > 2e-5:
            raise ValueError(f"Reset state differs from the reference: {reset_error}")
        if observations["obs"].shape != (count, 75) or observations["critic"].shape != (count, 570):
            raise ValueError("Runtime observations violate the full-21 contract")
        alpha = env.action_manager.get_term("joint_pos")._alpha
        default = mdp.default_q
        terminated = 0
        with torch.inference_mode():
            for _ in range(steps):
                current = env.state.info[SUPERMIMIC_STATE_KEY]
                target = mdp.motion.q[current["f"]]
                action = (target - default - current["xi"][:, 1:22]) / alpha
                state = env.step(action)
                if capacity is not None:
                    capacity.check()
                transition = state.info[SUPERMIMIC_TRANSITION_KEY]
                for name in ("s", "s1", "a", "xi", "r"):
                    if not torch.isfinite(transition[name]).all():
                        raise ValueError(f"Nonfinite runtime tensor: {name}")
                if transition["s"].shape != (count, 55) or transition["a"].shape != (count, 21):
                    raise ValueError("Runtime transition has incorrect dimensions")
                if transition["xi"].shape != (count, 25):
                    raise ValueError("Runtime simulation context has incorrect dimensions")
                terminated += int(state.terminated.sum())
        return dict(
            task=str(cfg.training.task_name),
            backend=backend,
            environments=count,
            physics_steps_per_action=round(cfg.env.ctrl_dt / cfg.env.sim_dt),
            control_steps=steps,
            reset_max_abs_error=reset_error,
            observation_dim=75,
            goal_dim=570,
            action_dim=21,
            state_dim=55,
            context_dim=25,
            termination_events=terminated,
            finite=True,
            physics_capacity=None if capacity is None else capacity.snapshot(),
            boundary="Reference PD runtime smoke only; no learned policy and no training",
        )
    finally:
        env.close()
        env.backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--all-clips", action="store_true", help="One environment per motion, all from frame zero"
    )
    parser.add_argument(
        "--task",
        choices=("lens110_super_mimic", "lens110_lafan1_super_mimic"),
        default="lens110_super_mimic",
    )
    args = parser.parse_args()
    result = check(args.backend, args.steps, args.task, args.all_clips)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
