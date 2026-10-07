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

from sim.val.scene import ARENAS, TABLE_SURFACES, arenas_with_base

from .model import EpisodeKey
from .store import EpisodeStore, atomic_write_json


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


BASE_ARENAS = ("living_room", "kitchen")


def _tuple3(value):
    value = tuple(float(x) for x in value)
    if len(value) != 3 or not all(np.isfinite(value)):
        raise ValueError("expected three finite values")
    return value


LIGHT_NAMES = ("key", "fill", "ceiling", "rim")
MAX_LIGHT_INTENSITY = 1.5


def _strip_none(value):
    """Drop optional (None) fields so configurations recorded before they existed keep their hashes."""
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_strip_none(v) for v in value]
    return value


def _unit3(value, name):
    value = _tuple3(value)
    if not all(0 <= x <= 1 for x in value):
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class FrontCamera:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float
    roll: float | None = None  # degrees about the viewing axis; None = level (legacy)

    def __post_init__(self):
        object.__setattr__(self, "pos", _tuple3(self.pos))
        object.__setattr__(self, "lookat", _tuple3(self.lookat))
        if not np.isfinite(self.fovy) or not 0 < self.fovy < 180:
            raise ValueError("front fovy must be between 0 and 180 degrees")
        if self.roll is not None and (not np.isfinite(self.roll) or abs(self.roll) > 30):
            raise ValueError("front roll must be within 30 degrees")
        direction = np.subtract(self.lookat, self.pos)
        if np.linalg.norm(direction[:2]) < 1e-9:
            raise ValueError("front camera cannot look straight vertically")


@dataclass(frozen=True)
class Light:
    name: str
    pos: tuple[float, float, float]
    intensity: float
    color: tuple[float, float, float]
    castshadow: bool | None = None  # None keeps the scene default (legacy)

    def __post_init__(self):
        object.__setattr__(self, "pos", _tuple3(self.pos))
        object.__setattr__(self, "color", _tuple3(self.color))
        if (self.name not in LIGHT_NAMES or not np.isfinite(self.intensity)
                or not 0 < self.intensity <= MAX_LIGHT_INTENSITY):
            raise ValueError("invalid light")
        if not all(0 <= x <= 1 for x in self.color):
            raise ValueError("light color must be in [0, 1]")
        if self.castshadow is not None and type(self.castshadow) is not bool:
            raise ValueError("castshadow must be a boolean")


@dataclass(frozen=True)
class Background:
    skybox_top: tuple[float, float, float]
    skybox_bottom: tuple[float, float, float]

    def __post_init__(self):
        for name in ("skybox_top", "skybox_bottom"):
            object.__setattr__(self, name, _unit3(getattr(self, name), "background color"))


@dataclass(frozen=True)
class VisualConfig:
    arena: str
    front_camera: FrontCamera
    lights: tuple[Light, ...]
    background: Background
    variation_seed: int
    resample_index: int = 0
    # Optional dimensions (None = scene default; omitted from the hash so legacy configs keep theirs).
    ambient: float | None = None
    table_surface: str | None = None
    table_tint: tuple[float, float, float] | None = None
    wall_tint: tuple[float, float, float] | None = None
    floor_tint: tuple[float, float, float] | None = None
    # Extra low clutter (names in sim.train.tasks.assets.EXTRA_DISTRACTORS) added to the family's pool, and
    # colours for unnamed-colour primitive task objects/mats (applied only when the instruction names no colour).
    distractor_extras: tuple[str, ...] | None = None
    primitive_palette: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self):
        if self.arena not in ARENAS:
            raise ValueError("arena is not qualified")
        object.__setattr__(self, "lights", tuple(self.lights))
        names = tuple(light.name for light in self.lights)
        if names[:2] != ("key", "fill") or len(set(names)) != len(names) or any(
                LIGHT_NAMES.index(a) > LIGHT_NAMES.index(b) for a, b in zip(names, names[1:])):
            raise ValueError("configuration requires key and fill lights in order")
        if type(self.variation_seed) is not int or self.variation_seed < 0:
            raise ValueError("variation_seed must be a nonnegative integer")
        if type(self.resample_index) is not int or self.resample_index < 0:
            raise ValueError("resample_index must be a nonnegative integer")
        if self.ambient is not None and (not np.isfinite(self.ambient) or not 0 <= self.ambient <= 0.6):
            raise ValueError("ambient must be in [0, 0.6]")
        if self.table_surface is not None and self.table_surface not in TABLE_SURFACES:
            raise ValueError("unknown table surface")
        for name in ("table_tint", "wall_tint", "floor_tint"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _unit3(getattr(self, name), name))
        if self.distractor_extras is not None:
            object.__setattr__(self, "distractor_extras", tuple(self.distractor_extras))
            if (len(set(self.distractor_extras)) != len(self.distractor_extras)
                    or not all(isinstance(n, str) and n.replace("_", "").isalnum() for n in self.distractor_extras)):
                raise ValueError("distractor_extras must be distinct asset names")
        if self.primitive_palette is not None:
            object.__setattr__(self, "primitive_palette",
                               tuple(_unit3(c, "primitive colour") for c in self.primitive_palette))

    @property
    def config_hash(self) -> str:
        # Exclude provenance: identical visuals must collide even across seeds.
        value = asdict(self)
        value.pop("variation_seed")
        value.pop("resample_index")
        return _hash(_strip_none(value))

    def to_dict(self) -> dict:
        return json.loads(json.dumps({**asdict(self), "config_hash": self.config_hash}))

    @classmethod
    def from_dict(cls, value: dict) -> VisualConfig:
        value = dict(value)
        expected = value.pop("config_hash")
        for name in ("distractor_extras", "primitive_palette"):
            if value.get(name) is not None:
                value[name] = tuple(tuple(x) if isinstance(x, list) else x for x in value[name])
        config = cls(**{**value, "front_camera": FrontCamera(**value["front_camera"]),
                        "lights": tuple(Light(**light) for light in value["lights"]),
                        "background": Background(**value["background"])})
        if config.config_hash != expected:
            raise ValueError("visual configuration hash mismatch")
        return config


def kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate black-body colour (Tanner Helland fit), normalised so the brightest channel is 1."""
    t = float(kelvin) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    rgb = np.clip([r, g, b], 0, 255)
    return tuple(float(x) for x in rgb / rgb.max())


# The rooms' walls (world frame, robot base at the origin): rear wall x ~ -0.7, side walls y = +-1.5, front
# wall x ~ +1.8, wall tops z ~ +1.1. Lights stay inside so no wall shadows the workspace.
LIGHT_BOX = ((-0.55, 1.6), (-1.3, 1.3), (0.45, 2.0))
LIGHT_TARGET = (0.18, 0.0, 0.0)  # scene.py aims every spot light here


def _pair(value):
    value = tuple(float(x) for x in value)
    if len(value) != 2 or not all(np.isfinite(value)) or value[0] > value[1]:
        raise ValueError("invalid bounds")
    return value


@dataclass(frozen=True)
class VariationPolicy:
    # Arena names; a base geometry name ("kitchen", "living_room") expands to every room built on that table.
    arenas: tuple[str, ...] = ("living_room", "kitchen")
    table_surfaces: tuple[str, ...] = tuple(TABLE_SURFACES)
    # Front camera: spherical about a sampled look-at point. Azimuth 0 = camera on +x looking back at the
    # robot; the half-width of the view at the look-at point fixes distance = half_width / tan(fovy / 2).
    front_lookat: tuple = ((0.12, 0.20), (-0.045, 0.035), (-0.01, 0.03))
    front_azimuth: tuple = (-62.0, 28.0)
    front_elevation: tuple = (28.0, 62.0)
    front_half_width: tuple = (0.22, 0.32)
    front_fovy: tuple = (40.0, 60.0)
    front_roll: tuple = (-6.0, 6.0)
    # Lights: key spot from any azimuth above the table, colour temperature in kelvin.
    key_azimuth: tuple = (-180.0, 180.0)
    key_elevation: tuple = (30.0, 80.0)
    key_distance: tuple = (1.0, 1.9)
    key_intensity: tuple = (0.3, 1.1)
    key_kelvin: tuple = (2700.0, 7500.0)
    key_shadow_probability: float = 0.8
    fill_azimuth_offset: tuple = (120.0, 240.0)  # relative to the key
    fill_elevation: tuple = (20.0, 60.0)
    fill_intensity: tuple = (0.08, 0.45)
    fill_kelvin: tuple = (3000.0, 9000.0)
    ceiling_intensity: tuple = (0.05, 0.35)
    ceiling_kelvin: tuple = (3500.0, 6500.0)
    rim_probability: float = 0.45
    rim_intensity: tuple = (0.15, 0.55)
    rim_kelvin: tuple = (2700.0, 8000.0)
    ambient: tuple = (0.08, 0.30)
    background_brightness: tuple = (0.5, 1.0)
    table_tint: tuple = (0.65, 1.0)   # brightness; hue jitter +-0.08 per channel
    wall_tint: tuple = (0.6, 1.0)
    floor_tint: tuple = (0.6, 1.0)
    # Extra distractors eligible for this task (TaskDefinition.variation_policy fills it) and how many of them
    # join the scene's pool per episode; palette size for primitive recolouring.
    extra_distractors: tuple[str, ...] = ()
    n_extra_distractors: int = 6
    palette_size: int = 6
    # Initial calibration: 100 pixels per 2.8-cm block at 640x480; pilot
    # qualification must tune this policy for smaller task-specific objects.
    min_object_pixels: int = 100
    min_goal_visible_fraction: float = 0.6
    frame_margin_pixels: int = 2
    # Exposure of the robot-free first frame (Rec. 601 luma, 0-255).
    exposure_mean_luma: tuple = (60.0, 190.0)
    max_clipped_fraction: float = 0.03   # luma >= 250
    max_dark_fraction: float = 0.25      # luma < 20

    def __post_init__(self):
        object.__setattr__(self, "arenas", expand_arenas(self.arenas))
        object.__setattr__(self, "table_surfaces", tuple(self.table_surfaces))
        object.__setattr__(self, "extra_distractors", tuple(self.extra_distractors))
        if len(set(self.extra_distractors)) != len(self.extra_distractors) or self.n_extra_distractors < 0 \
                or self.palette_size < 0:
            raise ValueError("invalid distractor/palette policy")
        if not self.table_surfaces or any(name not in TABLE_SURFACES for name in self.table_surfaces):
            raise ValueError("policy contains an unknown table surface")
        object.__setattr__(self, "front_lookat", tuple(_pair(p) for p in self.front_lookat))
        if len(self.front_lookat) != 3:
            raise ValueError("position policy requires three bounds")
        for name in ("front_azimuth", "front_elevation", "front_half_width", "front_fovy", "front_roll",
                     "key_azimuth", "key_elevation", "key_distance", "key_intensity", "key_kelvin",
                     "fill_azimuth_offset", "fill_elevation", "fill_intensity", "fill_kelvin", "ceiling_intensity",
                     "ceiling_kelvin", "rim_intensity", "rim_kelvin", "ambient", "background_brightness",
                     "table_tint", "wall_tint", "floor_tint", "exposure_mean_luma"):
            try:
                object.__setattr__(self, name, _pair(getattr(self, name)))
            except ValueError:
                raise ValueError(f"invalid bounds: {name}") from None
        if not 0 < self.front_fovy[0] <= self.front_fovy[1] < 180:
            raise ValueError("invalid fovy bounds")
        if not 5 <= self.front_elevation[0] <= self.front_elevation[1] <= 85:
            raise ValueError("invalid elevation bounds")
        if self.front_half_width[0] <= 0 or abs(self.front_roll[0]) > 30 or abs(self.front_roll[1]) > 30:
            raise ValueError("invalid camera bounds")
        if not 5 <= self.key_elevation[0] <= self.key_elevation[1] <= 90 or self.key_distance[0] <= 0:
            raise ValueError("invalid key light bounds")
        for name in ("key_kelvin", "fill_kelvin", "ceiling_kelvin", "rim_kelvin"):
            if not 1500 <= getattr(self, name)[0] <= getattr(self, name)[1] <= 12000:
                raise ValueError(f"invalid bounds: {name}")
        for name in ("key_intensity", "fill_intensity", "ceiling_intensity", "rim_intensity"):
            low, high = getattr(self, name)
            if not 0 < low <= high <= MAX_LIGHT_INTENSITY:
                raise ValueError(f"invalid bounds: {name}")
        for name in ("background_brightness", "table_tint", "wall_tint", "floor_tint"):
            low, high = getattr(self, name)
            if not 0 < low <= high <= 1:
                raise ValueError(f"invalid bounds: {name}")
        if not 0 <= self.ambient[0] <= self.ambient[1] <= 0.6:
            raise ValueError("invalid bounds: ambient")
        if not (0 <= self.key_shadow_probability <= 1 and 0 <= self.rim_probability <= 1):
            raise ValueError("probabilities must be in [0, 1]")
        if type(self.min_object_pixels) is not int or self.min_object_pixels < 1:
            raise ValueError("min_object_pixels must be positive")
        if not 0 < self.min_goal_visible_fraction <= 1 or self.frame_margin_pixels < 0:
            raise ValueError("invalid visibility thresholds")
        if not (0 <= self.exposure_mean_luma[0] < self.exposure_mean_luma[1] <= 255
                and 0 <= self.max_clipped_fraction <= 1 and 0 <= self.max_dark_fraction <= 1):
            raise ValueError("invalid exposure thresholds")


def expand_arenas(names) -> tuple[str, ...]:
    """Arena names with base geometries ("kitchen", "living_room") expanded to all rooms on that table."""
    names = tuple(names)
    if not names or any(name not in ARENAS for name in names):
        raise ValueError("policy contains an unqualified arena")
    out = []
    for name in names:
        for arena in (arenas_with_base(name) if name in BASE_ARENAS else (name,)):
            if arena not in out:
                out.append(arena)
    return tuple(out)


DEFAULT_POLICY = VariationPolicy()
# Tasks whose instruction is phrased in world axes as seen by the viewer (left = -y, front = +x, "faces the
# camera" = along x) keep the front camera near the original band (azimuth -36..-7, elevation 39..58 deg) and nearly
# level; a low camera also hides a goal spot "behind" its reference.
VIEW_ALIGNED = dict(front_azimuth=(-36.0, 12.0), front_elevation=(38.0, 62.0), front_roll=(-3.0, 3.0))


def _spherical(center, azimuth, elevation, distance):
    a, e = np.radians(azimuth), np.radians(elevation)
    return np.asarray(center, float) + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _light_pos(rng, azimuth, elevation, distance):
    """Spot position toward (azimuth, elevation) from the light target, pulled inside LIGHT_BOX."""
    direction = _spherical((0, 0, 0), azimuth, elevation, 1.0)
    target = np.asarray(LIGHT_TARGET)
    limit = distance
    for axis, (low, high) in enumerate(LIGHT_BOX):
        if direction[axis] > 1e-9:
            limit = min(limit, (high - target[axis]) / direction[axis])
        elif direction[axis] < -1e-9:
            limit = min(limit, (low - target[axis]) / direction[axis])
    pos = target + direction * max(limit, 0.5)
    return tuple(float(x) for x in np.clip(pos, [b[0] for b in LIGHT_BOX], [b[1] for b in LIGHT_BOX]))


def sample_visual_config(task: str, seed: int, policy: VariationPolicy = DEFAULT_POLICY,
                         *, resample_index: int = 0) -> VisualConfig:
    EpisodeKey(task, seed)
    if type(resample_index) is not int or resample_index < 0:
        raise ValueError("resample_index must be a nonnegative integer")
    digest = _hash({"task": task, "episode_seed": seed, "resample_index": resample_index})
    variation_seed = int(digest, 16)
    rng = np.random.default_rng(variation_seed)
    u = lambda bounds: float(rng.uniform(*bounds))

    def tint(bounds, jitter=0.08):  # brightness times per-channel hue jitter, never clipped
        level = u(bounds) / (1 + jitter)
        return tuple(float(level * (1 + rng.uniform(-jitter, jitter))) for _ in range(3))

    def kelvin(bounds):  # uniform in mired (perceptually even between warm and cool)
        return kelvin_rgb(1e6 / rng.uniform(1e6 / bounds[1], 1e6 / bounds[0]))

    arena = policy.arenas[int(rng.integers(len(policy.arenas)))]
    surface = policy.table_surfaces[int(rng.integers(len(policy.table_surfaces)))]
    lookat = tuple(u(b) for b in policy.front_lookat)
    fovy = u(policy.front_fovy)
    distance = u(policy.front_half_width) / np.tan(np.radians(fovy) / 2)
    pos = _spherical(lookat, u(policy.front_azimuth), u(policy.front_elevation), distance)
    camera = FrontCamera(tuple(float(x) for x in pos), lookat, fovy, u(policy.front_roll))
    key_azimuth = u(policy.key_azimuth)
    lights = [Light("key", _light_pos(rng, key_azimuth, u(policy.key_elevation), u(policy.key_distance)),
                    u(policy.key_intensity), kelvin(policy.key_kelvin),
                    bool(rng.random() < policy.key_shadow_probability)),
              Light("fill", _light_pos(rng, key_azimuth + u(policy.fill_azimuth_offset), u(policy.fill_elevation),
                                       1.5), u(policy.fill_intensity), kelvin(policy.fill_kelvin), False),
              Light("ceiling", _light_pos(rng, u((-180, 180)), u((60, 90)), 1.5), u(policy.ceiling_intensity),
                    kelvin(policy.ceiling_kelvin), False)]
    if rng.random() < policy.rim_probability:
        lights.append(Light("rim", _light_pos(rng, u((-180, 180)), u((25, 70)), 1.4), u(policy.rim_intensity),
                            kelvin(policy.rim_kelvin), bool(rng.random() < 0.3)))
    brightness = u(policy.background_brightness)
    background = Background(tuple(brightness * x for x in (0.9, 0.9, 1.0)),
                            tuple(brightness * x for x in (0.2, 0.3, 0.4)))
    extras = None
    if policy.extra_distractors and policy.n_extra_distractors:
        count = min(policy.n_extra_distractors, len(policy.extra_distractors))
        extras = tuple(policy.extra_distractors[i] for i in rng.choice(len(policy.extra_distractors), count,
                                                                         replace=False))
    palette = tuple(_primitive_colour(rng) for _ in range(policy.palette_size)) or None
    return VisualConfig(arena, camera, tuple(lights), background, variation_seed, resample_index,
                        ambient=u(policy.ambient), table_surface=surface, table_tint=tint(policy.table_tint),
                        wall_tint=tint(policy.wall_tint, 0.12), floor_tint=tint(policy.floor_tint),
                        distractor_extras=extras, primitive_palette=palette)


def _primitive_colour(rng):
    """A saturated, mid-to-bright colour (HSV) for a primitive block/mat."""
    import colorsys
    return tuple(float(x) for x in colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.45, 0.95),
                                                        rng.uniform(0.45, 0.95)))


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
    exposure: dict | None = None  # robot-free first-frame luma statistics

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def exposure_stats(rgb) -> dict:
    """Rec. 601 luma statistics of an RGB frame: mean, clipped (>= 250) and dark (< 20) pixel fractions."""
    rgb = np.asarray(rgb, dtype=np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # no BLAS matmul (spurious FP warnings)
    return {"mean_luma": round(float(luma.mean()), 3), "clipped_fraction": round(float((luma >= 250).mean()), 5),
            "dark_fraction": round(float((luma < 20).mean()), 5)}


def exposure_reasons(stats: dict, policy) -> list[str]:
    reasons = []
    low, high = policy.exposure_mean_luma
    if stats["mean_luma"] < low or stats["dark_fraction"] > policy.max_dark_fraction:
        reasons.append("underexposed")
    if stats["mean_luma"] > high or stats["clipped_fraction"] > policy.max_clipped_fraction:
        reasons.append("overexposed")
    return reasons



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
    exposure = exposure_stats(env.render_scene_without_robot("front"))
    reasons += exposure_reasons(exposure, policy)
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
    return ScreeningReport(not reasons, tuple(reasons), tuple(areas), fraction, config.resample_index + 1, exposure)


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
