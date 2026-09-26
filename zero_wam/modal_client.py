"""Run upstream RoboTwin evaluation against an ephemeral Modal GPU class."""

from __future__ import annotations

import os

from pathlib import Path
LATENT_PATH = (
    "/models/human_latents/robotwin/"
    "run_robotwin_20260728_013720_robotwin_n1000/samples/"
    "273_robotwin_Use_an_arm_to_place_the_empty_cup_on_the_coaster/"
    "generated_video_kling-v3.pth"
)
VIDEO_PATH = LATENT_PATH.replace("/human_latents/", "/human_data/").replace(
    ".pth", ".mp4"
)


class ModalPolicy:
    def __init__(self, model, mode: str, port: int | None = None):
        del port
        self.mode = mode
        if self.mode not in {"text", "latent", "video"}:
            raise ValueError("ZERO_WAM_MODE must be text, latent, or video")
        self.latent_path = os.environ.get("ZERO_WAM_ICL_LATENT_PATH", LATENT_PATH)
        self.video_path = os.environ.get("ZERO_WAM_ICL_VIDEO_PATH", VIDEO_PATH)
        self.model = model

    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):
            obs = dict(obs)
            obs["use_icl"] = self.mode != "text"
            obs["video_guidance_scale"] = 5.0 if self.mode == "text" else -1.0
            obs["icl_guidance_scale"] = 5.0
            obs["icl_latent_path"] = self.latent_path if self.mode == "latent" else ""
            obs["icl_video_path"] = self.video_path if self.mode == "video" else ""
        result = self.model.step.remote(obs)
        if "action" in result and result["action"].shape != (16, 2, 16):
            raise RuntimeError(f"Unexpected Zero-WAM action shape: {result['action'].shape}")
        return result


def run(model, mode: str, task: str, test_num: int, seed: int, save_root: str) -> None:
    """Keep SAPIEN local while the injected class lives in `modal run`."""
    import yaml

    robotwin_root = Path(os.environ.get("ROBOTWIN_ROOT", "/workspace/Robotwin"))
    zero_wam_root = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    from evaluation.robotwin import eval_policy_client_openpi as client

    client.WebsocketClientPolicy = lambda port: ModalPolicy(model, mode, port)
    client.Sapien_TEST()
    with (robotwin_root / "policy/ACT/deploy_policy.yml").open() as file:
        args = yaml.safe_load(file)
    args.update(
        save_root=save_root,
        video_guidance_scale=5.0 if mode == "text" else -1.0,
        action_guidance_scale=1.0,
        icl_guidance_scale=5.0,
        icl_human_video_map=str(
            zero_wam_root / "evaluation/robotwin/robotwin_icl_human_videos.py"
        ),
        icl_latent_root=str(
            Path(__file__).resolve().parents[1] / "data/HumanGen/human_latents/robotwin"
        ),
        icl_seed=seed,
        port=8000,
        test_num=test_num,
        task_name=task,
        task_config="demo_clean",
        train_config_name=0,
        model_name=0,
        ckpt_setting=0,
        seed=seed,
        policy_name="ACT",
    )
    client.main(args)
