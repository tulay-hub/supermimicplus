"""Differentiable MJCF kinematics; G1 names remain the backwards-compatible default."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from unisupermimic.rl.algos.supermimic.geometry import axis_angle_to_mat, quat_to_mat

from .robot import G1_JOINT_NAMES, TRACKED_BODIES


class G1Kinematics:
    def __init__(
        self,
        mjcf_path: str,
        device: str | torch.device,
        tracked: Sequence[str] = TRACKED_BODIES,
        joint_names: Sequence[str] = G1_JOINT_NAMES,
    ):
        import mujoco

        model = mujoco.MjModel.from_xml_path(mjcf_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.device = torch.device(device)
        self.joint_names = tuple(joint_names)
        self.num_joints = len(self.joint_names)
        self.tracked_names = tuple(tracked)
        if not self.joint_names or len(set(self.joint_names)) != self.num_joints:
            raise ValueError("Kinematics requires nonempty, unique joint names")
        joint_name_to_idx = {name: index for index, name in enumerate(self.joint_names)}
        for name in self.joint_names:
            joint = model.joint(name)
            if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:  # pyright: ignore[reportAttributeAccessIssue]
                raise ValueError(f"Expected a hinge joint: {name}")
            if any(abs(float(value)) > 1e-9 for value in model.jnt_pos[joint.id]):
                raise ValueError(f"Joint {name} needs a nonzero-anchor kinematics adapter")
        bodies: list[str] = []
        parent: list[int] = []
        joint_idx: list[int] = []
        axes, offsets_p, offsets_r = [], [], []
        mj_to_local = {}
        for body_id in range(1, model.nbody):
            name = model.body(body_id).name
            count = int(model.body_jntnum[body_id])
            if count > 1:
                raise ValueError(f"Multiple joints on body {name} are unsupported")
            joint_id = int(model.body_jntadr[body_id]) if count else -1
            mj_to_local[body_id] = len(bodies)
            bodies.append(name)
            if count and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:  # pyright: ignore[reportAttributeAccessIssue]
                if len(bodies) != 1:
                    raise ValueError("Kinematics requires one floating root as the first body")
                parent.append(-1)
                joint_idx.append(-1)
            else:
                parent_id = model.body_parentid[body_id]
                if parent_id not in mj_to_local:
                    raise ValueError(f"Body {name} is outside the floating-root tree")
                parent.append(mj_to_local[parent_id])
                joint_idx.append(joint_name_to_idx[model.joint(joint_id).name] if count else -1)
            axes.append(model.jnt_axis[joint_id].tolist() if count else [0.0, 0.0, 0.0])
            offsets_p.append(model.body_pos[body_id].tolist())
            offsets_r.append(model.body_quat[body_id].tolist())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.tensor(offsets_p, dtype=torch.float32, device=self.device)
        self.off_R = quat_to_mat(torch.tensor(offsets_r, dtype=torch.float32, device=self.device))
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)

        lower = [0.0] * self.num_joints
        upper = [0.0] * self.num_joints
        for joint_id in range(model.njnt):
            name = model.joint(joint_id).name
            if name in joint_name_to_idx:
                index = joint_name_to_idx[name]
                lower[index] = float(model.jnt_range[joint_id][0])
                upper[index] = float(model.jnt_range[joint_id][1])
        self.q_lo = torch.tensor(lower, dtype=torch.float32, device=self.device)
        self.q_hi = torch.tensor(upper, dtype=torch.float32, device=self.device)

    def to(self, device: str | torch.device) -> G1Kinematics:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        return new

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        all_bodies: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked-body world poses and link-origin linear/angular velocities."""
        positions = [p]
        rotations = [R]
        linear_velocities = [v]
        angular_velocities = [w]
        off_p = self.off_p.to(p)
        off_R = self.off_R.to(R)
        axes = self.axes.to(q)
        for index in range(1, self.num_bodies):
            parent_index = self.parent[index]
            joint_index = self.joint_idx[index]
            parent_rotation = rotations[parent_index]
            position = positions[parent_index] + (
                parent_rotation @ off_p[index].unsqueeze(-1)
            ).squeeze(-1)
            offset_rotation = parent_rotation @ off_R[index]
            if joint_index < 0:
                rotation = offset_rotation
                angular_velocity = angular_velocities[parent_index]
            else:
                rotation = offset_rotation @ axis_angle_to_mat(axes[index], q[..., joint_index])
                axis_world = (offset_rotation @ axes[index].unsqueeze(-1)).squeeze(-1)
                angular_velocity = (
                    angular_velocities[parent_index]
                    + axis_world * qd[..., joint_index : joint_index + 1]
                )
            linear_velocity = linear_velocities[parent_index] + torch.cross(
                angular_velocities[parent_index],
                position - positions[parent_index],
                dim=-1,
            )
            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)
        indices = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[index] for index in indices], -2),
            torch.stack([rotations[index] for index in indices], -3),
            torch.stack([linear_velocities[index] for index in indices], -2),
            torch.stack([angular_velocities[index] for index in indices], -2),
        )
