"""Scripted grasp check for the exact reachable cube-lift evaluation scene."""

from __future__ import annotations

import numpy as np

from zero_wam.so101_eval import make_env


def _ik(env, target, seed):
    import mujoco

    model = env.model
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    dofs = [model.joint(name).dofadr[0] for name in names]
    body = model.body("gripper").id
    offset = np.array([.01, 0., -.08])
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(200):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rot = data.xmat[body].reshape(3, 3)
        point = data.xpos[body] + rot @ offset
        error = np.r_[target - point, np.cross(-rot[:, 2], [0, 0, -1])]
        mujoco.mj_jac(model, data, jacp, jacr, point, body)
        jac = np.vstack([jacp, jacr])[:, dofs]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
        for index, name in enumerate(names):
            joint = model.joint(name)
            address = joint.qposadr[0]
            data.qpos[address] = np.clip(data.qpos[address] + delta[index], *model.jnt_range[joint.id])
    return np.array([data.qpos[model.joint(name).qposadr[0]] for name in names]), float(np.linalg.norm(error[:3]))


def run(seed=100):
    env = make_env("joint", cameras=False)
    _, info = env.reset(seed=seed)
    cube = env.data.xpos[env.model.body("pick_slot_0").id].copy()
    model_seed = env.data.qpos.copy()
    for name, angle in zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], [np.arctan2(-cube[1], cube[0]), .6, -.4, 1.35, 0.]):
        model_seed[env.model.joint(name).qposadr[0]] = angle
    points = [cube + [0, 0, height] for height in (.05, .012, .10)]
    targets = []
    for point in points:
        arm, error = _ik(env, point, model_seed)
        targets.append((arm, error))
        for index, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
            model_seed[env.model.joint(name).qposadr[0]] = arm[index]

    def move(target, grip, steps):
        nonlocal info
        start = env._get_current_qpos().copy()
        goal = np.r_[target, grip]
        for fraction in np.linspace(0, 1, steps):
            _, _, _, _, info = env.step(start + fraction * (goal - start))

    move(targets[0][0], 1.4, 100)
    move(targets[1][0], 1.4, 70)
    move(targets[1][0], -.17, 70)
    move(targets[2][0], -.17, 100)
    result = {"seed": seed, "success": bool(info["success"]), "is_grasped": float(info["is_grasped"]), "lift_height": float(info["lift_height"]), "ik_position_errors_m": [error for _, error in targets], "cube_xyz": cube.tolist()}
    env.close()
    return result


if __name__ == "__main__":
    import json

    for episode_seed in (100, 101, 102):
        print(json.dumps(run(episode_seed)))
