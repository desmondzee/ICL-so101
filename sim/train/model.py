"""Typed records for immutable episode inputs and explicit review gates."""

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any


class EpisodeState(StrEnum):
    CANDIDATE = "candidate"
    RECORDED = "recorded"
    PHYSICS_APPROVED = "physics_approved"
    ROBOT_APPROVED = "robot_approved"
    HUMAN_SUBMITTED = "human_submitted"
    HUMAN_COMPLETE = "human_complete"  # Generation/download only; never acceptance.
    HUMAN_REVIEW_APPROVED = "human_review_approved"
    VERIFIER_APPROVED = "verifier_approved"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EpisodeKey:
    task: str
    seed: int

    def __post_init__(self):
        if not isinstance(self.task, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.task):
            raise ValueError("task must be a nonempty filesystem-safe identifier")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class EpisodeManifest:
    key: EpisodeKey
    config_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    manifest: EpisodeManifest
    state: EpisodeState
    history: tuple[dict[str, Any], ...] = ()
    human_attempts: tuple[dict[str, Any], ...] = ()
