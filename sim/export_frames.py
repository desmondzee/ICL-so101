import json
from pathlib import Path

import mujoco
from PIL import Image

from export_examples import OUT
from libero_basket_env import TARGETS, LiberoBasketEnv
from record_demos import closing_dirs
from scripted import PickPlace


def render_without_robot(env):
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    renderer = env._overhead_obs_renderer
    renderer.update_scene(env.data, camera=env._overhead_obs_cam, scene_option=option)
    return Image.fromarray(renderer.render())


def export():
    env = LiberoBasketEnv()
    for folder in sorted(Path(OUT).glob("episode_*")):
        info = json.loads((folder / "episode.json").read_text())
        frames = folder / "frames"
        frames.mkdir(exist_ok=True)
        env.reset(seed=info["seed"])
        render_without_robot(env).save(frames / "first.png")
        for action in PickPlace(env, TARGETS, closing_dirs(env)).actions():
            *_, step_info = env.step(action)
        assert step_info["success"], folder
        render_without_robot(env).save(frames / "last.png")
        print(frames)
    env.close()


if __name__ == "__main__":
    export()
