from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(HERE / "so101" / "scene_camera.xml"))

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
arm_dofs = [model.joint(n).dofadr[0] for n in ARM]
arm_act = [model.actuator(n).id for n in ARM]
grip_act = model.actuator("gripper").id
gripper = model.body("gripper").id
GRASP_OFFSET = np.array([0.010, 0.0, -0.080])


def grasp_point(d):
    return d.xpos[gripper] + d.xmat[gripper].reshape(3, 3) @ GRASP_OFFSET


def ik(target, seed, iters=200):
    d = mujoco.MjData(model)
    d.qpos[:] = seed
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        R = d.xmat[gripper].reshape(3, 3)
        approach = -R[:, 2]
        err = np.concatenate([target - grasp_point(d), np.cross(approach, [0, 0, -1])])
        mujoco.mj_jac(model, d, jacp, jacr, grasp_point(d), gripper)
        J = np.vstack([jacp, jacr])[:, arm_dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        for i, name in enumerate(ARM):
            adr = model.joint(name).qposadr[0]
            lo, hi = model.jnt_range[model.joint(name).id]
            d.qpos[adr] = np.clip(d.qpos[adr] + dq[i], lo, hi)
    return d.qpos.copy(), np.linalg.norm(err[:3])


def move(data, target, grip, waypoints=40, steps_per=50):
    start, seed = grasp_point(data).copy(), data.qpos.copy()
    data.ctrl[grip_act] = grip
    for a in np.linspace(0, 1, waypoints)[1:]:
        seed, err = ik(start + a * (target - start), seed)
        data.ctrl[arm_act] = [seed[model.joint(n).qposadr[0]] for n in ARM]
        for _ in range(steps_per):
            mujoco.mj_step(model, data)
    return err


def pick(cube_xy):
    data = mujoco.MjData(model)
    cube_adr = model.joint(model.body("cube").jntadr[0]).qposadr[0]
    data.qpos[cube_adr:cube_adr + 2] = cube_xy
    mujoco.mj_forward(model, data)
    cube = model.body("cube").id
    start = data.xpos[cube].copy()
    OPEN, CLOSED = 1.2, -0.17
    seed = data.qpos.copy()
    for name, q in zip(ARM, [np.arctan2(-start[1], start[0]), 0.6, -0.4, 1.35, 0.0]):
        seed[model.joint(name).qposadr[0]] = q
    ready, _ = ik(start + [0, 0, 0.05], seed)
    data.qpos[:] = ready
    data.qpos[model.joint("gripper").qposadr[0]] = OPEN
    data.ctrl[arm_act] = [ready[model.joint(n).qposadr[0]] for n in ARM]
    data.ctrl[grip_act] = OPEN
    mujoco.mj_forward(model, data)
    move(data, start + [0, 0, 0.012], OPEN)
    for _ in range(1000):
        data.ctrl[grip_act] = CLOSED
        mujoco.mj_step(model, data)
    move(data, start + [0, 0, 0.10], CLOSED)
    for _ in range(500):
        mujoco.mj_step(model, data)
    return data, data.xpos[cube][2] - start[2]


results = {}
for xy in [(0.25, 0.0), (0.20, 0.0), (0.30, 0.0), (0.22, 0.08), (0.25, -0.10)]:
    data, lift = pick(xy)
    results[xy] = lift
    print(f"cube at {xy}: lifted {lift * 1000:5.1f} mm, gripper q = {data.qpos[model.joint('gripper').qposadr[0]]:.3f}")
    if xy == (0.25, 0.0):
        renderer = mujoco.Renderer(model, 360, 640)
        frames = []
        for cam in ("front", "wrist_cam"):
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        Image.fromarray(np.concatenate(frames, axis=1)).save(HERE / "check_grasp.png")

failed = [xy for xy, lift in results.items() if lift < 0.07]
assert not failed, f"grasp failed at {failed}"
print("grasp OK")
