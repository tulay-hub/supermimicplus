"""Fail closed on physical contact/constraint truncation in the mjwarp backend."""

from __future__ import annotations

from typing import Any

import torch


class PhysicsCapacityGuard:
    def __init__(
        self,
        overflow,
        constraints,
        contacts,
        candidates,
        *,
        contact_capacity: int,
        constraint_capacity: int,
        ccd_capacity: int,
    ):
        self.overflow = overflow
        self.constraints = constraints
        self.contacts = contacts
        self.candidates = candidates
        self.contact_capacity = contact_capacity
        self.constraint_capacity = constraint_capacity
        self.ccd_capacity = ccd_capacity
        self.peak_contacts = 0
        self.peak_constraints = 0
        self.peak_candidates = 0
        self.checks = 0
        self.overflow_flags = 0

    @classmethod
    def from_env(cls, env: Any) -> PhysicsCapacityGuard | None:
        data = getattr(getattr(env, "backend", None), "_device_data", None)
        if data is None or not hasattr(data, "overflow"):
            return None
        import warp

        return cls(
            warp.to_torch(data.overflow),
            warp.to_torch(data.nefc),
            warp.to_torch(data.nacon),
            warp.to_torch(data.ncollision),
            contact_capacity=int(data.naconmax),
            constraint_capacity=int(data.njmax),
            ccd_capacity=int(data.naccdmax),
        )

    def check(self) -> None:
        flags, constraints, contacts, candidates = (
            torch.stack(
                (
                    self.overflow.max(),
                    self.constraints.max(),
                    self.contacts.reshape(-1)[0],
                    self.candidates.reshape(-1)[0],
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        self.peak_contacts = max(self.peak_contacts, int(contacts))
        self.peak_constraints = max(self.peak_constraints, int(constraints))
        self.peak_candidates = max(self.peak_candidates, int(candidates))
        self.checks += 1
        self.overflow_flags |= int(flags)
        if flags or contacts > self.contact_capacity or constraints > self.constraint_capacity:
            raise RuntimeError(
                f"mjwarp physics capacity overflow: flags={flags}, contacts={contacts}/"
                f"{self.contact_capacity}, constraints={constraints}/{self.constraint_capacity}"
            )

    def snapshot(self) -> dict[str, int]:
        return dict(
            contact_capacity_total=self.contact_capacity,
            constraint_capacity_per_world=self.constraint_capacity,
            ccd_capacity_total=self.ccd_capacity,
            peak_contacts_total=self.peak_contacts,
            peak_constraints_per_world=self.peak_constraints,
            peak_broadphase_candidates_total=self.peak_candidates,
            checked_control_steps=self.checks,
            overflow_flags=self.overflow_flags,
        )
