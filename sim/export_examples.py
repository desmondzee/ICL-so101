import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from record_demos import FEATURES, FPS, ROOT

OUT = Path(__file__).resolve().parents[1] / "data" / "examples" / ROOT.name


def export():
    ds = LeRobotDataset("local/so101_libero_basket", root=ROOT, video_backend="pyav")
    if OUT.exists():
        shutil.rmtree(OUT)
    summary = json.loads((ROOT / "demo_summary.json").read_text())
    for ep in range(ds.num_episodes):
        start, end = ds.meta.episodes["dataset_from_index"][ep], ds.meta.episodes["dataset_to_index"][ep]
        folder = OUT / f"episode_{ep:03d}"
        single = LeRobotDataset.create(
            repo_id=f"local/{ROOT.name}_episode_{ep:03d}",
            fps=FPS,
            features=FEATURES,
            root=folder,
            robot_type="so101_follower",
            use_videos=True,
            vcodec="h264",
        )
        for i in range(start, end):
            item = ds[i]
            frame = {
                k: (item[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8) if FEATURES[k]["dtype"] == "video" else item[k].numpy()
                for k in FEATURES
            }
            single.add_frame({**frame, "task": item["task"]})
        single.save_episode()
        single.finalize()
        (folder / "episode.json").write_text(json.dumps({"task": item["task"], "fps": FPS, "frames": end - start, **summary[ep]}, indent=2, default=bool))
        print(folder)


if __name__ == "__main__":
    export()
