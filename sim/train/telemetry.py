"""Lossless substep evidence, stored as compressed numeric arrays without pickle.

MuJoCo leaves contacts/forces from the solve *before* its final integration.
``time`` labels integrated qpos/qvel; ``contact_time`` labels that solver state.
We deliberately never call mj_forward on live data to refresh contacts: doing
so would replace the evidence and can change the subsequent simulation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import mujoco
import numpy as np


SCHEMA_VERSION = 1
ARRAY_FIELDS = frozenset((
    "time", "contact_time", "frame", "frame_end", "joint_qpos", "joint_qvel", "joint_target", "actuator_force",
    "qpos", "qvel", "qacc", "body_pose", "body_velocity", "success", "grasp", "warnings", "contact_offsets",
    "contact_geom", "contact_distance", "contact_force", "contact_normal", "contact_position",
))


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass
class Telemetry:
    arrays: dict[str, np.ndarray]
    manifest: dict

    def save(self, path: str | Path) -> None:
        """Embed the versioned manifest in the NPZ; never overwrite evidence."""
        payload = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if set(self.arrays) != ARRAY_FIELDS or any(a.dtype.kind not in "biuf" for a in self.arrays.values()):
            raise ValueError("telemetry arrays must be numeric and pickle-free")
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, manifest=np.array(payload), **self.arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Telemetry":
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"]),
                                  parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            return cls({key: archive[key].copy() for key in archive.files if key != "manifest"}, manifest)


class TelemetryCollector:
    """Create after reset; pass to every ``env.step(substep_observer=...)``.

    Body names include all free bodies (including parked distractors) and task
    fixtures. Contact owners collapse object/fixture child bodies to their
    declared root; robot links and arena bodies retain their model names, so
    task policies can distinguish fingers from arm/table/fixture collisions.
    The supplied config_hash identifies the complete episode, whereas
    visual_config_hash identifies its visual configuration alone.
    """

    def __init__(self, env, *, config_hash: str, visual_config_hash: str):
        if not _valid_hash(config_hash) or not _valid_hash(visual_config_hash):
            raise ValueError("config_hash and visual_config_hash must be explicit SHA-256 digests")
        self.env = env
        self.names = [*env.free_names, *env.fixture_names]
        model = env.model
        owners = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            owner = next((name for name in self.names if env._in_subtree(body, env._body[name])), None)
            owners.append(owner or model.body(body).name or f"body_{body}")
        self.manifest = dict(
            schema_version=SCHEMA_VERSION, config_hash=config_hash, visual_config_hash=visual_config_hash,
            body_names=self.names, free_body_names=list(env.free_names),
            active_distractors=list(env.active_distractors), fixture_names=list(env.fixture_names),
            geom_names=[model.geom(g).name or f"geom_{g}" for g in range(model.ngeom)],
            geom_bodies=owners, joint_names=[model.joint(j).name for j in env._joint_ids],
            joint_ranges=model.jnt_range[env._joint_ids].tolist(),
            timestep=float(model.opt.timestep), substeps=int(env._N_SUBSTEPS),
            units=dict(time="s", position="m", joint="rad", linear_velocity="m/s",
                       angular_velocity="rad/s", contact_force="N; torque N*m"),
            contact_state="pre-integration solver; see contact_time",
            control_mode=env.control_mode,
        )
        self.rows = []
        self.frame = 0
        self._capture(-1, -1)

    def __call__(self, env, substep: int) -> None:
        if env is not self.env:
            raise ValueError("collector cannot mix environments")
        self._capture(self.frame, substep)
        if substep == env._N_SUBSTEPS - 1:
            self.frame += 1

    def _capture(self, frame, substep):
        env, model, data = self.env, self.env.model, self.env.data
        poses, velocities = [], []
        for name in self.names:
            if name in env._qadr:
                a, v = env._qadr[name], env._dadr[name]
                poses.append(data.qpos[a:a + 7].copy())
                # Free-joint angular velocity is local. Rotate into world axes
                # before finite differencing for acceleration during the audit.
                velocity = data.qvel[v:v + 6].copy()
                mujoco.mju_rotVecQuat(velocity[3:], velocity[3:].copy(), data.qpos[a + 3:a + 7])
                velocities.append(velocity)
            else:
                poses.append(env.object_pose(name))
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, env._body[name], velocity, 0)
                velocities.append(np.r_[velocity[3:], velocity[:3]])
        geoms, distance, forces, normals, positions = [], [], [], [], []
        for i, contact in enumerate(data.contact[:data.ncon]):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            geoms.append([int(contact.geom1), int(contact.geom2)])
            distance.append(float(contact.dist))
            forces.append(force)
            normals.append(contact.frame[:3].copy())
            positions.append(contact.pos.copy())
        # Grasp readers change the upstream object's contact mask. Restore it
        # even if a task success reader fails; telemetry is observational.
        original_target = getattr(env, "_obj_geom_ids", None)
        original_mask = env._obj_geom_mask.copy()
        try:
            success = bool(env.success())
            grasp = [env.is_grasping(name) if name in env._qadr else False for name in self.names]
        finally:
            env._obj_geom_mask = original_mask
            if original_target is None:
                if hasattr(env, "_obj_geom_ids"):
                    del env._obj_geom_ids
            else:
                env._obj_geom_ids = original_target
        self.rows.append(dict(
            time=float(data.time), contact_time=float(data.time - model.opt.timestep),
            frame=frame, frame_end=substep == env._N_SUBSTEPS - 1,
            joint_qpos=data.qpos[env._qpos_addrs].copy(), joint_qvel=data.qvel[env._qvel_addrs].copy(),
            joint_target=data.ctrl[env._actuator_ids].copy(), actuator_force=data.actuator_force[env._actuator_ids].copy(),
            qpos=data.qpos.copy(), qvel=data.qvel.copy(), qacc=data.qacc.copy(),
            body_pose=np.asarray(poses).reshape(-1, 7), body_velocity=np.asarray(velocities).reshape(-1, 6),
            success=success, grasp=grasp, warnings=np.asarray(data.warning.number).copy(),
            contact_geom=np.asarray(geoms, dtype=np.int32).reshape(-1, 2),
            contact_distance=np.asarray(distance), contact_force=np.asarray(forces).reshape(-1, 6),
            contact_normal=np.asarray(normals).reshape(-1, 3), contact_position=np.asarray(positions).reshape(-1, 3),
        ))

    def finish(self) -> Telemetry:
        contact_keys = ("contact_geom", "contact_distance", "contact_force", "contact_normal", "contact_position")
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0] if key not in contact_keys}
        arrays.update({key: np.concatenate([row[key] for row in self.rows], axis=0) for key in contact_keys})
        arrays["contact_offsets"] = np.r_[0, np.cumsum([len(row["contact_geom"]) for row in self.rows])]
        return Telemetry(arrays, json.loads(json.dumps(self.manifest)))
