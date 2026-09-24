import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from record_demos import FPS, ROOT


def export():
    ds = LeRobotDataset("local/so101_libero_basket", root=ROOT, video_backend="pyav")
    out_dir = ROOT / "previews"
    out_dir.mkdir(exist_ok=True)
    for ep in range(ds.num_episodes):
        start, end = ds.meta.episodes["dataset_from_index"][ep], ds.meta.episodes["dataset_to_index"][ep]
        path = out_dir / f"episode_{ep:03d}.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.pix_fmt = "yuv420p"
            for i in range(start, end):
                item = ds[i]
                frame = np.concatenate(
                    [(item[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for k in ("observation.images.wrist", "observation.images.overhead")],
                    axis=1,
                )
                stream.width, stream.height = frame.shape[1], frame.shape[0]
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            container.mux(stream.encode())
        print(path)


if __name__ == "__main__":
    export()
