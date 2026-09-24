import mujoco
import numpy as np

from so101_nexus.kinematics import quat_to_rotvec

OPEN, RELEASE, CLOSED = 1.2, 0.6, -0.17
SPEED = 0.18
CARRY_Z = 0.15
FIXED_FACE = 0.0199
GRASP = {
    "alphabet_soup": dict(width=0.031, z=0.024, place=np.array([0.0, 0.012, 0.08])),
    "cream_cheese": dict(width=0.021, z=0.008, place=np.array([0.0, -0.017, 0.085])),
}


def top_down_rotvec(closing_dir):
    x = closing_dir / np.linalg.norm(closing_dir)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat_to_rotvec(quat)


class PickPlace:
    def __init__(self, env, order, closing):
        self.env = env
        self.order = list(order)
        self.closing = closing
        self.log = []
        self.step = SPEED * env.control_dt

    def _goto(self, target, rotvec, grip, max_steps=330, tol=0.004, label=""):
        pos = self.env._get_tcp_pose()[:3].copy() if self.cmd is None else self.cmd
        for i in range(max_steps):
            step = target - pos
            dist = np.linalg.norm(step)
            pos = target.copy() if dist < self.step else pos + step / dist * self.step
            self.cmd = pos
            yield np.concatenate([pos, rotvec, [grip]])
            err = np.linalg.norm(self.env._get_tcp_pose()[:3] - target)
            if dist < self.step and err < tol:
                break
        self.log.append((label, "goto", i + 1, round(float(err) * 1000, 1)))

    def _hold(self, rotvec, grip, seconds):
        for _ in range(round(seconds / self.env.control_dt)):
            yield np.concatenate([self.cmd, rotvec, [grip]])

    def actions(self):
        self.cmd = None
        basket = self.env.object_pos("basket")
        for name in self.order:
            g = GRASP[name]
            d = self.closing[name]
            rotvec = top_down_rotvec(d)
            for attempt in range(2):
                obj = self.env.object_pos(name)
                offset = -(FIXED_FACE - g["width"] / 2 - 0.003) * d / np.linalg.norm(d)
                grasp = np.array([obj[0], obj[1], g["z"]]) + offset
                yield from self._goto(grasp + [0, 0, 0.07], rotvec, OPEN, label="pregrasp")
                yield from self._goto(grasp, rotvec, OPEN, tol=0.003, label="grasp")
                yield from self._hold(rotvec, CLOSED, 0.8)
                grasped = self.env.is_grasping(name)
                self.log.append((name, "grasp", attempt, grasped))
                if grasped:
                    break
                yield from self._hold(rotvec, OPEN, 0.5)
                yield from self._goto(grasp + [0, 0, 0.07], rotvec, OPEN)
            yield from self._goto(np.array([grasp[0], grasp[1], CARRY_Z]), rotvec, CLOSED, label="lift")
            self.log.append((name, "lifted", self.env.is_grasping(name)))
            held = self.env._get_tcp_pose()[:3] - self.env.object_pos(name)
            place = basket + g["place"] + [held[0], held[1], 0]
            yield from self._goto(np.array([place[0], place[1], CARRY_Z]), rotvec, CLOSED, label="carry")
            yield from self._goto(place, rotvec, CLOSED, label="lower")
            yield from self._hold(rotvec, RELEASE, 0.7)
            yield from self._goto(np.array([place[0], place[1], CARRY_Z]), rotvec, RELEASE)
            self.log.append((name, "in_basket", self.env.in_basket(name)))
