"""Frozen, bounded visual variation and front-view admission evidence.

Task reset remains the sole owner of object layouts. Callers reset with their
episode seed, screen that settled layout, and persist both poses and this
configuration. Resampling only changes the visual configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .model import EpisodeKey
from .store import EpisodeStore, atomic_write_json


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tuple3(value):
    value = tuple(float(x) for x in value)
    if len(value) != 3 or not all(np.isfinite(value)):
        raise ValueError("expected three finite values")
    return value


@dataclass(frozen=True)
class FrontCamera:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float

    def __post_init__(self):
        object.__setattr__(self, "pos", _tuple3(self.pos))
        object.__setattr__(self, "lookat", _tuple3(self.lookat))
        if not np.isfinite(self.fovy) or not 0 < self.fovy < 180:
            raise ValueError("front fovy must be between 0 and 180 degrees")
        direction = np.subtract(self.lookat, self.pos)
        if np.linalg.norm(direction[:2]) < 1e-9:
            raise ValueError("front camera cannot look straight vertically")


@dataclass(frozen=True)
class Light:
    name: str
    pos: tuple[float, float, float]
    intensity: float
    color: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "pos", _tuple3(self.pos))
        object.__setattr__(self, "color", _tuple3(self.color))
        if self.name not in ("key", "fill") or not np.isfinite(self.intensity) or not 0 < self.intensity <= 1:
            raise ValueError("invalid key/fill light")
        if not all(0 <= x <= 1 for x in self.color):
            raise ValueError("light color must be in [0, 1]")


@dataclass(frozen=True)
class Background:
    skybox_top: tuple[float, float, float]
    skybox_bottom: tuple[float, float, float]

    def __post_init__(self):
        for name in ("skybox_top", "skybox_bottom"):
            object.__setattr__(self, name, _tuple3(getattr(self, name)))
            if not all(0 <= x <= 1 for x in getattr(self, name)):
                raise ValueError("background color must be in [0, 1]")


@dataclass(frozen=True)
class VisualConfig:
    arena: str
    front_camera: FrontCamera
    lights: tuple[Light, ...]
    background: Background
    variation_seed: int
    resample_index: int = 0

    def __post_init__(self):
        if self.arena not in ("living_room", "kitchen"):
            raise ValueError("arena is not qualified")
        object.__setattr__(self, "lights", tuple(self.lights))
        if tuple(light.name for light in self.lights) != ("key", "fill"):
            raise ValueError("configuration requires key and fill lights in order")
        if type(self.variation_seed) is not int or self.variation_seed < 0:
            raise ValueError("variation_seed must be a nonnegative integer")
        if type(self.resample_index) is not int or self.resample_index < 0:
            raise ValueError("resample_index must be a nonnegative integer")

    @property
    def config_hash(self) -> str:
        # Exclude provenance: identical visuals must collide even across seeds.
        value = asdict(self)
        value.pop("variation_seed")
        value.pop("resample_index")
        return _hash(value)

    def to_dict(self) -> dict:
        return json.loads(json.dumps({**asdict(self), "config_hash": self.config_hash}))

    @classmethod
    def from_dict(cls, value: dict) -> VisualConfig:
        value = dict(value)
        expected = value.pop("config_hash")
        config = cls(**{**value, "front_camera": FrontCamera(**value["front_camera"]),
                        "lights": tuple(Light(**light) for light in value["lights"]),
                        "background": Background(**value["background"])})
        if config.config_hash != expected:
            raise ValueError("visual configuration hash mismatch")
        return config


@dataclass(frozen=True)
class VariationPolicy:
    arenas: tuple[str, ...] = ("living_room", "kitchen")
    front_pos: tuple = ((0.43, 0.53), (-0.19, -0.07), (0.34, 0.44))
    front_lookat: tuple = ((0.13, 0.19), (-0.035, 0.015), (-0.01, 0.025))
    front_fovy: tuple = (44.0, 54.0)
    key_pos: tuple = ((0.55, 0.85), (0.55, 1.05), (1.4, 1.9))
    fill_pos: tuple = ((-0.6, -0.2), (-1.4, -0.9), (1.1, 1.6))
    key_intensity: tuple = (0.48, 0.72)
    fill_intensity: tuple = (0.23, 0.36)
    warmth: tuple = (0.0, 0.22)
    background_brightness: tuple = (0.88, 1.0)
    # Initial calibration: 100 pixels per 2.8-cm block at 640x480; pilot
    # qualification must tune this policy for smaller task-specific objects.
    min_object_pixels: int = 100
    min_goal_visible_fraction: float = 0.6
    frame_margin_pixels: int = 2

    def __post_init__(self):
        object.__setattr__(self, "arenas", tuple(self.arenas))
        if not self.arenas or any(name not in ("living_room", "kitchen") for name in self.arenas):
            raise ValueError("policy contains an unqualified arena")
        for name in ("front_pos", "front_lookat", "key_pos", "fill_pos"):
            values = tuple(tuple(float(x) for x in pair) for pair in getattr(self, name))
            if len(values) != 3:
                raise ValueError("position policy requires three bounds")
            object.__setattr__(self, name, values)
        for name in ("front_fovy", "key_intensity", "fill_intensity", "warmth", "background_brightness"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("front_pos", "front_lookat", "key_pos", "fill_pos", "front_fovy",
                     "key_intensity", "fill_intensity", "warmth", "background_brightness"):
            value = getattr(self, name)
            pairs = value if name.endswith("pos") or name == "front_lookat" else (value,)
            for pair in pairs:
                if len(pair) != 2 or not all(np.isfinite(pair)) or pair[0] > pair[1]:
                    raise ValueError(f"invalid bounds: {name}")
        if not 0 < self.front_fovy[0] <= self.front_fovy[1] < 180:
            raise ValueError("invalid fovy bounds")
        if not 0 <= self.warmth[0] <= self.warmth[1] <= 0.22:
            raise ValueError("light warmth must be neutral to warm")
        for name in ("key_intensity", "fill_intensity", "background_brightness"):
            low, high = getattr(self, name)
            if not 0 < low <= high <= 1:
                raise ValueError(f"invalid bounds: {name}")
        if type(self.min_object_pixels) is not int or self.min_object_pixels < 1:
            raise ValueError("min_object_pixels must be positive")
        if not 0 < self.min_goal_visible_fraction <= 1 or self.frame_margin_pixels < 0:
            raise ValueError("invalid visibility thresholds")


DEFAULT_POLICY = VariationPolicy()


def sample_visual_config(task: str, seed: int, policy: VariationPolicy = DEFAULT_POLICY,
                         *, resample_index: int = 0) -> VisualConfig:
    EpisodeKey(task, seed)
    if type(resample_index) is not int or resample_index < 0:
        raise ValueError("resample_index must be a nonnegative integer")
    digest = _hash({"task": task, "episode_seed": seed, "resample_index": resample_index})
    variation_seed = int(digest, 16)
    rng = np.random.default_rng(variation_seed)

    def position(bounds):
        return tuple(float(rng.uniform(low, high)) for low, high in bounds)

    arena = policy.arenas[int(rng.integers(len(policy.arenas)))]
    camera = FrontCamera(position(policy.front_pos), position(policy.front_lookat),
                         float(rng.uniform(*policy.front_fovy)))
    lights = []
    for name, bounds, intensity in (("key", policy.key_pos, policy.key_intensity),
                                    ("fill", policy.fill_pos, policy.fill_intensity)):
        pos = position(bounds)
        level = float(rng.uniform(*intensity))
        warmth = float(rng.uniform(*policy.warmth))
        lights.append(Light(name, pos, level, (1.0, 1 - 0.15 * warmth, 1 - 0.35 * warmth)))
    brightness = float(rng.uniform(*policy.background_brightness))
    background = Background(tuple(brightness * x for x in (0.9, 0.9, 1.0)),
                            tuple(brightness * x for x in (0.2, 0.3, 0.4)))
    return VisualConfig(arena, camera, tuple(lights), background, variation_seed, resample_index)


def episode_config_hash(config: VisualConfig, object_poses: dict) -> str:
    """Hash the visuals together with the settled layout supplied by task reset."""
    if not object_poses:
        raise ValueError("episode identity requires task-owned object poses")
    return _hash({"visual_config_hash": config.config_hash, "object_poses": object_poses})


@dataclass(frozen=True)
class GoalRegion:
    """World-space samples covering a task's goal surface (not its centre alone).

    body names the supporting goal object, or None for an empty spatial goal.
    Samples should lie on that goal surface; the task owns their definition.
    """
    body: str | None
    points: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        object.__setattr__(self, "points", tuple(_tuple3(point) for point in self.points))
        if not self.points:
            raise ValueError("goal region needs surface samples")


@dataclass(frozen=True)
class ScreeningReport:
    accepted: bool
    reasons: tuple[str, ...]
    object_pixels: tuple[tuple[str, int], ...]
    goal_visible_fraction: float
    next_resample_index: int

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def _visual_geoms(env, name):
    root = env.model.body(name).id
    return [i for i in range(env.model.ngeom)
            if env._in_subtree(env.model.geom_bodyid[i], root)
            and env.model.geom_group[i] in (0, 1, 2)
            and env.model.geom_rgba[i, 3] > 0]


def _project(env, points, width, height):
    camera = env._front_cam_id
    local = (np.asarray(points) - env.data.cam_xpos[camera]) @ env.data.cam_xmat[camera].reshape(3, 3)
    depth = -local[:, 2]
    focal = height / (2 * np.tan(np.radians(env.model.cam_fovy[camera]) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.column_stack((width / 2 + focal * local[:, 0] / depth,
                              height / 2 - focal * local[:, 1] / depth))
    return xy, depth


def _inside(env, points, width, height, margin):
    xy, depth = _project(env, points, width, height)
    return (np.isfinite(xy).all(axis=1) & (depth > 0)
            & (xy[:, 0] >= margin) & (xy[:, 0] < width - margin)
            & (xy[:, 1] >= margin) & (xy[:, 1] < height - margin))


def _geom_corners(env, geoms):
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    points = []
    for geom in geoms:
        box = env.model.geom_aabb[geom]
        points.extend((box[:3] + corners * box[3:]) @ env.data.geom_xmat[geom].reshape(3, 3).T
                      + env.data.geom_xpos[geom])
    return points


def _goal_fraction(env, region, width, height, margin):
    camera = env.data.cam_xpos[env._front_cam_id]
    inside = _inside(env, region.points, width, height, margin)
    if region.body is not None:
        env.model.body(region.body)  # Fail loudly on a misspelled support name.
    visible = 0
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    for point, in_frame in zip(region.points, inside):
        if not in_frame:
            continue
        vector = np.subtract(point, camera)
        distance = float(np.linalg.norm(vector))
        if distance < 1e-9:
            continue
        hit = mujoco.mj_ray(env.model, env.data, camera, vector / distance, groups, 1, -1, geom_id)
        if hit < 0 or hit >= distance - 0.002:
            visible += 1
    return visible / len(region.points)


def _duplicate_in_store(store, config_hash):
    for path in sorted((store.root / "candidates").glob("*/episode_*/manifest.json")):
        # Store.load validates immutable bytes and fails closed on corruption.
        value = json.loads(path.read_text())
        record = store.load(EpisodeKey(**value["key"]))
        if record.manifest.visual_config_hash == config_hash:
            return True
    return False


def screen_visual_config(config: VisualConfig, env, task_objects: tuple[str, ...],
                         goal_regions: tuple[GoalRegion, ...], *, store: EpisodeStore | None = None,
                         policy: VariationPolicy = DEFAULT_POLICY) -> ScreeningReport:
    """Screen the current settled environment with bounds, segmentation and rays.

    Includes robot/distractor occlusion. Never moves objects or samples layouts.
    Call again at the terminal pose: an initial screen cannot certify a rollout.
    """
    if not task_objects or len(set(task_objects)) != len(task_objects):
        raise ValueError("task_objects must contain distinct body names")
    mujoco.mj_forward(env.model, env.data)
    renderer = env._renderer_rgb
    width, height = renderer.width, renderer.height
    reasons = []
    front = config.front_camera
    camera = env._front_cam_id
    if (env.visual_config != config or env.scene.arena != config.arena
            or not np.allclose(env.model.cam_pos[camera], front.pos, rtol=1e-5, atol=1e-6)
            or not np.isclose(env.model.cam_fovy[camera], front.fovy)):
        reasons.append("config_mismatch")
    areas = []
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(env.data, camera="front")
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    for name in task_objects:
        geoms = _visual_geoms(env, name)
        points = _geom_corners(env, geoms)
        if not points or not _inside(env, points, width, height, policy.frame_margin_pixels).all():
            if "object_outside_frame" not in reasons:
                reasons.append("object_outside_frame")
        pixels = int(np.count_nonzero(np.isin(segmentation[:, :, 0], geoms)
                                     & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)))
        areas.append((name, pixels))
        if pixels < policy.min_object_pixels and "object_too_small" not in reasons:
            reasons.append("object_too_small")
    fraction = min((_goal_fraction(env, region, width, height, policy.frame_margin_pixels)
                    for region in goal_regions), default=0.0)
    if not goal_regions:
        reasons.append("missing_goal_region")
    elif fraction < policy.min_goal_visible_fraction:
        reasons.append("goal_occluded")
    if store is not None and _duplicate_in_store(store, config.config_hash):
        reasons.append("duplicate_config")
    return ScreeningReport(not reasons, tuple(reasons), tuple(areas), fraction, config.resample_index + 1)


def save_screening_report(store: EpisodeStore, key: EpisodeKey, config: VisualConfig,
                          report: ScreeningReport) -> Path:
    """Keep immutable rejected attempts, including the deterministic retry index."""
    path = store.root / "variation_screening" / key.task / f"episode_{key.seed}" / f"attempt_{config.resample_index}.json"
    value = {"schema_version": 1, "key": asdict(key), "visual_config": config.to_dict(),
             "report": report.to_dict()}
    try:
        atomic_write_json(path, value, overwrite=False)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise ValueError("screening attempt already exists with different evidence")
    return path
