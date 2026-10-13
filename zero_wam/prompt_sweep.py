"""Pure configuration and result helpers for controlled SO-101 prompt sweeps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
NULL_PROMPT = " "  # Truthy for upstream reset; negative embedding is selected below.
TASK_PROMPTS = {
    "soup_lift": "Pick up the alphabet soup can.",
    "cheese_lift": "Pick up the cream cheese box.",
}
TASK_OBJECTS = {
    "soup_lift": "alphabet_soup",
    "cheese_lift": "cream_cheese",
}
OPPOSITE_TASK = {"soup_lift": "cheese_lift", "cheese_lift": "soup_lift"}


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    run: str
    step: int
    path: str


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    text_task: str | None
    video_task: str | None
    shuffle_video: bool = False

    @property
    def use_text(self) -> bool:
        return self.text_task is not None

    @property
    def use_video(self) -> bool:
        return self.video_task is not None

    def reset_payload(self, video_path: str | None = None) -> dict[str, Any]:
        if self.use_video != bool(video_path):
            raise ValueError(
                f"{self.name}: video path presence must match video conditioning"
            )
        return {
            "prompt": TASK_PROMPTS[self.text_task] if self.use_text else NULL_PROMPT,
            "use_icl": self.use_video,
            "icl_video_path": video_path or "",
            "icl_latent_path": "",
            # A negative target-text scale selects the learned empty embedding.
            "video_guidance_scale": 5.0 if self.use_text else -1.0,
            "icl_guidance_scale": 5.0 if self.use_video else 1.0,
        }


def prompt_conditions(target_task: str) -> list[ConditionSpec]:
    if target_task not in TASK_PROMPTS:
        raise ValueError(f"unsupported target task: {target_task}")
    other = OPPOSITE_TASK[target_task]
    return [
        ConditionSpec("no_conditioning", None, None),
        ConditionSpec("correct_text", target_task, None),
        ConditionSpec("incorrect_text", other, None),
        ConditionSpec("correct_human_video", None, target_task),
        ConditionSpec("mismatched_human_video", None, other),
        ConditionSpec("shuffled_human_video", None, target_task, True),
        ConditionSpec("matching_text_video", target_task, target_task),
        ConditionSpec("conflicting_text_video", target_task, other),
    ]


def validate_condition(spec: ConditionSpec, target_task: str) -> None:
    expected = {condition.name: condition for condition in prompt_conditions(target_task)}
    if expected.get(spec.name) != spec:
        raise ValueError(f"non-canonical condition for {target_task}: {spec}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(spec: CheckpointSpec) -> dict[str, Any]:
    root = Path(spec.path).resolve()
    transformer = root / "transformer" if (root / "transformer").is_dir() else root
    config = transformer / "config.json"
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise FileNotFoundError(f"incomplete transformer checkpoint: {transformer}")
    if "zero-wam-posttrain-robotwin" in str(transformer).lower():
        raise ValueError("RoboTwin checkpoint is forbidden for the SO-101 sweep")
    shards = sorted(transformer.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"checkpoint has no safetensor shards: {transformer}")
    return {
        "name": spec.name,
        "run": spec.run,
        "step": spec.step,
        "path": str(root),
        "transformer_path": str(transformer),
        "config_sha256": sha256_file(config),
        "index_sha256": sha256_file(index),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in shards
        ],
    }


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def condition_manifest(
    *,
    checkpoint: dict[str, Any],
    target_task: str,
    condition: ConditionSpec,
    environment_seed: int,
    sample_seed: int,
    initial_state_sha256: str,
    initial_observation_sha256: str,
    video: dict[str, Any] | None,
    max_controls: int,
) -> dict[str, Any]:
    validate_condition(condition, target_task)
    payload = condition.reset_payload(video["path"] if video else None)
    manifest = {
        "schema": SCHEMA_VERSION,
        "protocol": "paired_closed_loop_so101_prompt_sweep",
        "checkpoint": checkpoint,
        "target_task": target_task,
        "target_object": TASK_OBJECTS[target_task],
        "condition": asdict(condition),
        "reset_conditioning": payload,
        "environment_seed": environment_seed,
        "sample_seed": sample_seed,
        "initial_state_sha256": initial_state_sha256,
        "initial_observation_sha256": initial_observation_sha256,
        "human_video": video,
        "max_controls": max_controls,
        "closed_loop": {
            "first_chunk_skips_first_action_frame": True,
            "observation_keyframe_stride": 2,
            "kv_cache_action_history": "executed_lerobot_joint_targets",
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def load_results(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).glob("**/result.json")):
        row = json.loads(path.read_text())
        row["_result_path"] = str(path)
        rows.append(row)
    return rows


def render_report(root: str | Path, rows: list[dict[str, Any]]) -> str:
    root = Path(root)
    lines = [
        "# Zero-WAM -> SO-101 controlled prompt sweep",
        "",
        "This report separates action execution, prompt sensitivity, and evidence for",
        "in-context learning. A video-conditioned success alone is not sufficient:",
        "the paired correct/mismatched/shuffled controls must switch behavior while",
        "environment and sampling inputs remain fixed.",
        "",
        "## Rollouts",
        "",
        "| Checkpoint | Task | Seed | Condition | Selected object | Target success | Grasp | Controls | Rollout |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        selected = row.get("selected_object") or "-"
        link = Path(row["_result_path"]).parent / "rollout.mp4"
        try:
            link = link.relative_to(root)
        except ValueError:
            pass
        lines.append(
            f"| {row['checkpoint']} | {row['target_task']} | {row['environment_seed']} "
            f"| {row['condition']} | {selected} | {int(row['task_success'])} "
            f"| {int(row['target_grasp_success'])} | {row['controls']} "
            f"| [{link}]({link}) |"
        )
    paired = {}
    for row in rows:
        key = (row["checkpoint"], row["target_task"], row["environment_seed"])
        paired.setdefault(key, {})[row["condition"]] = row
    valid_pairs = [
        group for group in paired.values()
        if {"correct_text", "incorrect_text", "correct_human_video",
            "mismatched_human_video", "shuffled_human_video"} <= group.keys()
    ]
    text_switch = sum(
        group["correct_text"].get("selected_object") == group["correct_text"]["target_object"]
        and group["incorrect_text"].get("selected_object")
        == TASK_OBJECTS[OPPOSITE_TASK[group["correct_text"]["target_task"]]]
        for group in valid_pairs
    )
    video_switch = sum(
        group["correct_human_video"].get("selected_object")
        == group["correct_human_video"]["target_object"]
        and group["mismatched_human_video"].get("selected_object")
        == TASK_OBJECTS[OPPOSITE_TASK[group["correct_human_video"]["target_task"]]]
        for group in valid_pairs
    )
    lines += [
        "",
        "## Causal comparisons",
        "",
        f"- Complete paired cohorts: {len(valid_pairs)}",
        f"- Text prompt-switch accuracy: {text_switch}/{len(valid_pairs)}"
        if valid_pairs else "- Text prompt-switch accuracy: not measured",
        f"- Human-video prompt-switch accuracy: {video_switch}/{len(valid_pairs)}"
        if valid_pairs else "- Human-video prompt-switch accuracy: not measured",
        "",
        "## Interpretation",
        "",
        "- **Action prediction/execution:** evaluate target grasp and task success independently of prompt contrasts.",
        "- **Prompt sensitivity:** requires behavior to differ in the intended direction under paired prompt changes.",
        "- **Genuine ICL:** requires correct video to outperform mismatched and temporally shuffled video with text removed; text-only effects do not establish ICL.",
        "",
        "## Failures and limitations",
        "",
    ]
    failures = [row for row in rows if row.get("error")]
    lines.append(
        "- No recorded rollout errors." if not failures
        else f"- {len(failures)} rollout errors are recorded in result artifacts."
    )
    if not rows:
        lines.append("- No rollouts have been executed; all conclusions remain pending.")
    return "\n".join(lines) + "\n"
