"""Compare frozen recorded layouts and first camera frames to replay resets."""
from pathlib import Path
import json
import cv2
import numpy as np

from scripts.run_so101_batched_train_replay import DATA, TrainSim, load_cohort
from zero_wam.train_scene_replay import compare_recorded_layout, reset_recorded_scene
from zero_wam.so101_closed_loop import FRONT, WRIST
from zero_wam.prompt_sweep import atomic_json


def main():
    out = Path("outputs/zero_wam/train_replay_scene_audit")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in load_cohort():
        sim = TrainSim(ep["task"], ep["seed"], ep["visual_config"])
        try:
            before = sim.observe()
            before_check = compare_recorded_layout(sim.env.object_poses(), ep["object_poses_start"])
            restoration = reset_recorded_scene(sim.env, ep["seed"], ep["object_poses_start"])
            after = sim.observe()
            images, metrics = [], {}
            for name, key in (("front", FRONT), ("wrist", WRIST)):
                cap = cv2.VideoCapture(str(DATA / "episodes" / ep["id"] / f"robot_{name}.mp4"))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise ValueError(f"missing training frame: {ep['id']} {name}")
                original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if original.shape != after[key].shape:
                    raise ValueError("camera shape mismatch")
                metrics[name] = {
                    "before_pixel_mae": float(np.abs(original.astype(float)-before[key]).mean()),
                    "after_pixel_mae": float(np.abs(original.astype(float)-after[key]).mean()),
                }
                panels = []
                for label, image in (("recorded", original), ("old replay", before[key]),
                                     ("corrected replay", after[key])):
                    panel = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                    cv2.putText(panel, f"{name}: {label}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    panels.append(panel)
                images.append(np.concatenate(panels, axis=1))
            image_name = ep["id"].replace("/", "__") + ".jpg"
            cv2.imwrite(str(out / image_name), np.concatenate(images, axis=0))
            results.append({
                "id": ep["id"], "seed": ep["seed"], "before": before_check,
                "restoration": restoration, "visual_config_hash": ep["visual_config_hash"],
                "live_visual_config_hash": sim.env.visual_config.config_hash,
                "camera_pixel_comparison": metrics, "comparison_image": image_name,
            })
            print(ep["id"], restoration["sampler"], restoration["check"]["max_abs_pose_error"], flush=True)
        finally:
            sim.env.close()
    atomic_json(out / "audit.json", {
        "cause_commit": "7f8ce421409236af076af3fc8b33db77523e0452",
        "note": "Pixel MAE includes MP4 compression and first-frame timing; no pixel threshold used.",
        "episodes": results,
    })


if __name__ == "__main__":
    main()
