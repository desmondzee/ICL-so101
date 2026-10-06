"""MJCF builder for the SO-101 validation scenes.

World frame = robot base frame: the robot base sits at the origin facing +x, and the arena is shifted so the
table top is z = 0. LIBERO meshes are loaded at a per-object scale (0.5 by default, matching sim/libero_scene.py),
their collision boxes are hidden (group 3), and visuals are group 1. The robot's visual geoms are the only group-2
geoms in the model, which is what `ValEnv.render_scene_without_robot` hides.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from so101_nexus import get_so101_mujoco_model_path
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "third_party/LIBERO/libero/libero/assets"
ARENA_SCALE = 0.5


@dataclass(frozen=True)
class Asset:
    path: str
    upright: tuple = ()
    density: float = 500.0


H = (0, np.pi / 2)
CATALOG = {
    "plate": Asset("stable_scanned_objects/plate/plate.xml", density=1200),
    "white_bowl": Asset("stable_scanned_objects/white_bowl/white_bowl.xml", density=600),
    "akita_black_bowl": Asset("stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", density=600),
    "red_bowl": Asset("stable_scanned_objects/red_bowl/red_bowl.xml", density=600),
    "ramekin": Asset("stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml", density=900),
    "frypan": Asset("stable_scanned_objects/chefmate_8_frypan/chefmate_8_frypan.xml", density=700),
    "basket": Asset("stable_scanned_objects/basket/basket.xml", density=150),
    "porcelain_mug": Asset("turbosquid_objects/porcelain_mug/porcelain_mug.xml", density=500),
    "red_coffee_mug": Asset("turbosquid_objects/red_coffee_mug/red_coffee_mug.xml", density=500),
    "white_yellow_mug": Asset("turbosquid_objects/white_yellow_mug/white_yellow_mug.xml", density=500),
    "black_book": Asset("turbosquid_objects/black_book/black_book.xml", density=600),
    "yellow_book": Asset("turbosquid_objects/yellow_book/yellow_book.xml", density=600),
    "moka_pot": Asset("turbosquid_objects/moka_pot/moka_pot.xml", density=500),
    "alphabet_soup": Asset("stable_hope_objects/alphabet_soup/alphabet_soup.xml", (H,), 900),
    "tomato_sauce": Asset("stable_hope_objects/tomato_sauce/tomato_sauce.xml", (H,), 900),
    "cream_cheese": Asset("stable_hope_objects/cream_cheese/cream_cheese.xml", (), 900),
    "butter": Asset("stable_hope_objects/butter/butter.xml", (), 900),
    "chocolate_pudding": Asset("stable_hope_objects/chocolate_pudding/chocolate_pudding.xml", (), 900),
    "popcorn": Asset("stable_hope_objects/popcorn/popcorn.xml", (), 300),
    "ketchup": Asset("stable_hope_objects/ketchup/ketchup.xml", (H, (2, np.pi / 2)), 700),
    "milk": Asset("stable_hope_objects/milk/milk.xml", (H, (2, np.pi / 2)), 700),
    "orange_juice": Asset("stable_hope_objects/orange_juice/orange_juice.xml", (H, (2, np.pi / 2)), 700),
    "bbq_sauce": Asset("stable_hope_objects/bbq_sauce/bbq_sauce.xml", (), 700),
    "microwave": Asset("articulated_objects/microwave.xml"),
    "flat_stove": Asset("articulated_objects/flat_stove.xml"),
}

ARENAS = {
    "living_room": dict(xml="scenes/libero_living_room_tabletop_base_style.xml", robot_xy=(-0.26, 0.0), probe_xy=(-0.1, 0.0), table=None),
    "kitchen": dict(xml="scenes/libero_kitchen_tabletop_base_style.xml", robot_xy=(-0.30, 0.0), probe_xy=(0.0, 0.0), table=(0.35, 0.5)),
}

ROBOT_RGBA = (0.93, 0.93, 0.91, 1.0)

VISUAL_XML = """
<visual>
  <headlight diffuse="0 0 0" ambient="0.22 0.22 0.22" specular="0 0 0"/>
  <quality shadowsize="8192" offsamples="8"/>
  <map znear="0.004" zfar="40" haze="0.15" shadowscale="0.6"/>
  <global offwidth="1280" offheight="960"/>
</visual>"""

LIGHTS_XML = """
<light name="key" pos="0.7 0.8 1.6" dir="-0.4 -0.45 -0.8" directional="false" cutoff="45" exponent="1"
       diffuse="0.6 0.58 0.55" specular="0.2 0.2 0.2" castshadow="true" bulbradius="0.08"/>
<light name="fill" pos="-0.4 -1.2 1.3" dir="0.25 0.75 -0.6" directional="false" cutoff="60" exponent="1"
       diffuse="0.28 0.3 0.33" specular="0.05 0.05 0.05" castshadow="false"/>
<light name="ceiling" pos="0 0 2.0" dir="0 0 -1" directional="true" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"
       castshadow="false"/>"""


@dataclass
class Obj:
    """A free-floating LIBERO object. `rgba` multiplies its visual materials (tint)."""

    name: str
    asset: str
    scale: float = 0.5
    rgba: tuple | None = None
    mass: float | None = None
    friction: float = 1.0


@dataclass
class Block:
    """A free primitive box; `half` is the half-extent in metres."""

    name: str
    half: tuple = (0.014, 0.014, 0.014)
    rgba: tuple = (0.8, 0.1, 0.1, 1.0)
    mass: float = 0.03
    friction: float = 1.2


@dataclass
class Disc:
    """A free flat cylinder (coaster/mat). `radius`, `half_height` in metres."""

    name: str
    radius: float = 0.05
    half_height: float = 0.003
    rgba: tuple = (0.2, 0.3, 0.8, 1.0)
    mass: float = 0.05
    friction: float = 1.0


@dataclass
class Fixture:
    """A static LIBERO object (no free joint); move it per reset with `ValEnv.set_fixture_pose`.
    Articulated joints inside (microwave door) are kept and named `<name>_<joint>`."""

    name: str
    asset: str
    scale: float = 0.5


@dataclass
class SceneSpec:
    arena: str = "living_room"
    objects: list = field(default_factory=list)
    fixtures: list = field(default_factory=list)
    distractors: list = field(default_factory=list)
    robot_rgba: tuple = ROBOT_RGBA

    @property
    def free_bodies(self):
        return [*self.objects, *self.distractors]


def _vec(el, key, default=None):
    return np.array(el.get(key).split(), dtype=float) if key in el.attrib else default


def _set(el, key, v):
    el.set(key, " ".join(f"{x:.6g}" for x in np.atleast_1d(v)))


def _upright_quat(steps):
    q = np.array([1.0, 0, 0, 0])
    for axis, angle in steps:
        r = np.zeros(4)
        mujoco.mju_axisAngle2Quat(r, np.eye(3)[axis], angle)
        mujoco.mju_mulQuat(q, r, q.copy())
    return q


def _fragment(xml):
    return list(ET.fromstring(f"<root>{xml}</root>"))


def load_mjcf(path: Path, scale: float, prefix: str | None = None):
    """Parse a LIBERO MJCF, make file paths absolute, scale it, hide group-0 collision geoms, prefix names."""
    root = ET.parse(path).getroot()
    for el in root.iter():
        if "file" in el.attrib and not el.get("file").startswith("/"):
            el.set("file", str((path.parent / el.get("file")).resolve()))
        if el.tag in ("body", "geom", "site", "camera", "light", "inertial", "joint") and "pos" in el.attrib:
            _set(el, "pos", _vec(el, "pos") * scale)
        if el.tag in ("geom", "site") and "size" in el.attrib and el.get("type") != "mesh":
            _set(el, "size", _vec(el, "size") * scale)
        if el.tag == "mesh":
            _set(el, "scale", _vec(el, "scale", np.ones(3)) * scale)
        if el.tag in ("inertial", "geom") and "mass" in el.attrib:
            el.set("mass", f"{float(el.get('mass')) * scale**3:.6g}")
        if el.tag == "geom":
            el.attrib.pop("solref", None)
            el.attrib.pop("solimp", None)
            if el.get("group") == "0":
                el.set("group", "3")
        if prefix:
            for key in ("name", "mesh", "material", "texture", "class", "childclass"):
                if key in el.attrib:
                    el.set(key, f"{prefix}_{el.get(key)}")
    return root


def _collision_geoms(root):
    return [g for g in root.iter("geom") if g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0"]


def _hull_volume(root):
    mj = ET.Element("mujoco")
    mj.append(copy.deepcopy(root.find("asset")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    vols = []
    for i in range(model.nmesh):
        a, n = model.mesh_vertadr[i], model.mesh_vertnum[i]
        if n >= 4:
            vols.append(ConvexHull(model.mesh_vert[a : a + n]).volume)
    return max(vols)


def _set_free_physics(root, mass, friction):
    boxes = [g for g in _collision_geoms(root) if g.get("type") == "box"]
    volume = sum(8 * np.prod(_vec(g, "size")) for g in boxes)
    for g in root.iter("geom"):
        if g not in boxes:
            g.set("density", "0")
            g.attrib.pop("mass", None)
    for g in boxes:
        g.attrib.update(density=f"{mass / volume:.6g}", condim="4", friction=f"{friction} 0.02 0.001", group="3")


def _soften(asset_el, max_reflectance=0.05):
    for m in asset_el.iter("material"):
        if float(m.get("reflectance", 0)) > max_reflectance:
            m.set("reflectance", str(max_reflectance))


def _tint(asset_el, rgba):
    for m in asset_el.iter("material"):
        base = _vec(m, "rgba", np.ones(4))
        _set(m, "rgba", base * np.asarray(rgba))


def _content(obj_root):
    """Children of the object's `object` body (and its sites) to put under our own body."""
    outer = obj_root.find("worldbody/body")
    inner = next(b for b in outer.iter("body") if b.get("name", "").endswith("object"))
    return list(inner) + [s for s in outer if s.tag == "site"]


def _add_assets(asset_el, obj_root):
    seen = {(el.tag, el.get("name")) for el in asset_el}
    for el in obj_root.find("asset"):
        if (el.tag, el.get("name")) not in seen:
            asset_el.append(el)
            seen.add((el.tag, el.get("name")))


def _merge_defaults(mj, obj_root):
    defaults = obj_root.find("default")
    if defaults is None:
        return
    top = mj.find("default")
    if top is None:
        top = ET.Element("default")
        mj.insert(2, top)
    top.extend(list(defaults))


def _free_body(mj, asset_el, worldbody, spec):
    if hasattr(spec, "build_mjcf"):  # extension point (sim.train.tasks.assets.Scanned); unused by validation
        spec.build_mjcf(mj, asset_el, worldbody)
        return
    if isinstance(spec, Block):
        body = ET.SubElement(worldbody, "body", name=spec.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{spec.name}_joint")
        ET.SubElement(body, "geom", name=f"{spec.name}_geom", type="box", size=" ".join(map(str, spec.half)),
                      rgba=" ".join(map(str, spec.rgba)), mass=str(spec.mass), group="1", condim="4",
                      friction=f"{spec.friction} 0.02 0.001", material="val_plastic")
        return
    if isinstance(spec, Disc):
        body = ET.SubElement(worldbody, "body", name=spec.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{spec.name}_joint")
        ET.SubElement(body, "geom", name=f"{spec.name}_geom", type="cylinder", size=f"{spec.radius} {spec.half_height}",
                      rgba=" ".join(map(str, spec.rgba)), mass=str(spec.mass), group="1", condim="4",
                      friction=f"{spec.friction} 0.02 0.001", material="val_fabric")
        return
    a = CATALOG[spec.asset]
    root = load_mjcf(ASSETS / a.path, spec.scale, spec.name)
    _soften(root.find("asset"))
    if spec.rgba is not None:
        _tint(root.find("asset"), spec.rgba)
    mass = spec.mass if spec.mass is not None else _hull_volume(root) * a.density
    _set_free_physics(root, mass, spec.friction)
    _merge_defaults(mj, root)
    _add_assets(asset_el, root)
    body = ET.SubElement(worldbody, "body", name=spec.name, pos="0 0 -1")
    ET.SubElement(body, "freejoint", name=f"{spec.name}_joint")
    upright = ET.SubElement(body, "body", name=f"{spec.name}_upright")
    _set(upright, "quat", _upright_quat(a.upright))
    upright.extend(_content(root))


def _fixture(mj, asset_el, worldbody, spec):
    a = CATALOG[spec.asset]
    root = load_mjcf(ASSETS / a.path, spec.scale, spec.name)
    _merge_defaults(mj, root)
    _add_assets(asset_el, root)
    for j in root.iter("joint"):
        if j.get("type") == "free":
            j.set("type", "hinge")
            j.set("range", "0 0")
    body = ET.SubElement(worldbody, "body", name=spec.name, pos="0 0 -1")
    upright = ET.SubElement(body, "body", name=f"{spec.name}_upright")
    _set(upright, "quat", _upright_quat(a.upright))
    upright.extend(_content(root))


def _arena(name):
    cfg = ARENAS[name]
    root = load_mjcf(ASSETS / cfg["xml"], ARENA_SCALE)
    _soften(root.find("asset"))
    if cfg["table"] is not None:
        for g in root.iter("geom"):
            if g.get("name") in ("table_collision", "table_visual"):
                size = _vec(g, "size")
                size[:2] = cfg["table"]
                _set(g, "size", size)
            if g.get("name", "").startswith("table_leg"):
                g.set("rgba", "0 0 0 0")
    for g in root.iter("geom"):
        if g.get("group") in (None, "2"):
            g.set("group", "1")
    return root


def table_top_z(arena: str) -> float:
    cfg = ARENAS[arena]
    root = _arena(arena)
    mj = ET.Element("mujoco")
    mj.append(root.find("asset"))
    wb = ET.SubElement(mj, "worldbody")
    wb.extend(copy.deepcopy([el for el in root.find("worldbody") if el.tag != "light"]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(model, data, np.array([*cfg["probe_xy"], 5.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    return 5.0 - d


def camera_xml(name, pos, lookat, fovy):
    """A fixed camera at `pos` looking at `lookat` with image-up roughly world +z."""
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    fwd = (lookat - pos) / np.linalg.norm(lookat - pos)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    xy = " ".join(f"{v:.6g}" for v in (*right, *up))
    return f'<camera name="{name}" mode="fixed" pos="{" ".join(f"{v:.6g}" for v in pos)}" xyaxes="{xy}" fovy="{fovy}"/>'


def build_scene_xml(spec: SceneSpec, cameras: list[str] = (), *, visual_config=None) -> str:
    """MJCF string for `spec`; `cameras` are extra <camera> elements placed in the worldbody."""
    arena_name = spec.arena if visual_config is None else visual_config.arena
    cfg = ARENAS[arena_name]
    arena = _arena(arena_name)
    mj = ET.Element("mujoco", model=f"so101_val_{arena_name}")
    ET.SubElement(mj, "include", file=str(get_so101_mujoco_model_path()))
    mj.extend(_fragment(MUJOCO_SCENE_OPTION_XML + VISUAL_XML))
    asset = ET.SubElement(mj, "asset")
    asset.extend(_fragment(
        '<material name="val_plastic" specular="0.15" shininess="0.25" reflectance="0"/>'
        '<material name="val_fabric" specular="0.02" shininess="0.0" reflectance="0"/>'))
    worldbody = ET.SubElement(mj, "worldbody")
    world = ET.SubElement(worldbody, "body", name="arena")
    _set(world, "pos", [-cfg["robot_xy"][0], -cfg["robot_xy"][1], -table_top_z(arena_name)])
    if visual_config is not None:
        for texture in arena.find("asset").iter("texture"):
            if texture.get("type") == "skybox":
                _set(texture, "rgb1", visual_config.background.skybox_top)
                _set(texture, "rgb2", visual_config.background.skybox_bottom)
    asset.extend(arena.find("asset"))
    world.extend(el for el in arena.find("worldbody") if el.tag not in ("camera", "light"))
    worldbody.extend(_fragment(LIGHTS_XML + "".join(cameras)))
    if visual_config is not None:
        for camera in list(worldbody.findall("camera")):
            if camera.get("name") == "front":
                worldbody.remove(camera)
        front = visual_config.front_camera
        worldbody.extend(_fragment(camera_xml("front", front.pos, front.lookat, front.fovy)))
        for sampled in visual_config.lights:
            light = worldbody.find(f"light[@name='{sampled.name}']")
            _set(light, "pos", sampled.pos)
            _set(light, "dir", np.array([0.18, 0.0, 0.0]) - sampled.pos)
            _set(light, "diffuse", np.array(sampled.color) * sampled.intensity)
    for spec_obj in spec.free_bodies:
        _free_body(mj, asset, worldbody, spec_obj)
    for fx in spec.fixtures:
        _fixture(mj, asset, worldbody, fx)
    return ET.tostring(mj, encoding="unicode")
