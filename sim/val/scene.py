"""MJCF builder for the SO-101 validation scenes.

World frame = robot base frame: the robot base sits at the origin facing +x, and the arena is shifted so the
table top is z = 0. LIBERO meshes are loaded at a per-object scale (0.5 by default, matching sim/libero_scene.py),
their collision boxes are hidden (group 3), and visuals are group 1. The robot's visual geoms are the only group-2
geoms in the model, which is what `ValEnv.render_scene_without_robot` hides.
"""

from __future__ import annotations

import copy
import functools
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
    # Training-only additions (validation never references them).
    "cookies": Asset("stable_hope_objects/cookies/cookies.xml", (), 300),
    "macaroni_and_cheese": Asset("stable_hope_objects/macaroni_and_cheese/macaroni_and_cheese.xml", (), 400),
    "salad_dressing": Asset("stable_hope_objects/salad_dressing/salad_dressing.xml", (), 700),
    "new_salad_dressing": Asset("stable_hope_objects/new_salad_dressing/new_salad_dressing.xml", (), 700),
    "black_book_flat": Asset("turbosquid_objects/black_book/black_book.xml", ((1, np.pi / 2),), 600),
    "yellow_book_flat": Asset("turbosquid_objects/yellow_book/yellow_book.xml", ((1, np.pi / 2),), 600),
    "microwave": Asset("articulated_objects/microwave.xml"),
    "flat_stove": Asset("articulated_objects/flat_stove.xml"),
}

ARENAS = {
    "living_room": dict(xml="scenes/libero_living_room_tabletop_base_style.xml", robot_xy=(-0.26, 0.0), probe_xy=(-0.1, 0.0), table=None),
    "kitchen": dict(xml="scenes/libero_kitchen_tabletop_base_style.xml", robot_xy=(-0.30, 0.0), probe_xy=(0.0, 0.0), table=(0.35, 0.5)),
}

# ----- training rooms: visual-only restyles of the two qualified table geometries -----------------------------
# Validation only ever uses the two base arenas above (bit-identical XML). Training rooms reuse a base arena's
# table, floor plane and walls unchanged (physics identical: same collision geoms, robot offset and table top) and
# only swap floor/wall textures and the visual-only (contype 0) backdrop props along the rear/side walls.
_KITCHEN_XML = "scenes/libero_kitchen_tabletop_base_style.xml"
_LIVING_XML = "scenes/libero_living_room_tabletop_base_style.xml"
_STUDY_XML = "scenes/libero_study_base_style.xml"
# (source scene, mesh, pos override in arena coordinates after 0.5x scaling or None, quat override or None)
PROP_SETS = {
    "kitchen": tuple((_KITCHEN_XML, m, None, None) for m in (
        "kitchen_background_vis", "kitchen_background_stove_vis", "kitchen_background_hot_pot_vis",
        "kitchen_background_pot_vis", "kitchen_background_fridge_vis")),
    "living_room": ((_LIVING_XML, "living_room_vis", None, None), (_LIVING_XML, "wall_decoration_vis", None, None)),
    "study": ((_STUDY_XML, "study_wall_painting_vis", None, None), (_STUDY_XML, "office_book_shelf_vis", None, None),
              (_STUDY_XML, "plant_vis", None, None), (_STUDY_XML, "black_book_vis", None, None),
              (_STUDY_XML, "floor_lamp_vis", (-0.8, 0.55, 0.45), None)),
    "gallery": ((_LIVING_XML, "wall_decoration_vis", (-0.97, -0.35, 0.75), None),
                (_STUDY_XML, "study_wall_painting_vis", (-0.95, 0.45, 0.55), None),
                (_STUDY_XML, "plant_vis", (-0.85, 0.95, 0.0), None)),
    "shelves": ((_STUDY_XML, "office_book_shelf_vis", (-0.85, 0.35, 0.0), None),
                (_STUDY_XML, "office_book_shelf_vis", (-0.85, -0.45, 0.0), None),
                (_STUDY_XML, "plant_vis", (-0.85, -1.05, 0.0), None)),
    "none": (),
}
_T = "textures/"
ROOMS = {
    # name: base geometry, floor texture, wall texture, backdrop prop set
    "kitchen_dark": ("kitchen", "dark_floor_texture.png", "dark_gray_plaster.png", "kitchen"),
    "kitchen_bright": ("living_room", "white_marble_floor.png", "white_wall.png", "kitchen"),
    "kitchen_terracotta": ("kitchen", "brown_ceramic_tile.png", "cream-plaster.png", "kitchen"),
    "study": ("kitchen", "seamless_wood_planks_floor.png", "meeka-beige-plaster.png", "study"),
    "study_blue": ("living_room", "light-gray-floor-tile.png", "dark_blue_wall.png", "study"),
    "lounge_warm": ("living_room", "rustic_floor.png", "yellow_linen_wall_texture.png", "living_room"),
    "lounge_green": ("kitchen", "dapper_gray_floor.png", "dark_green_plaster_wall.png", "living_room"),
    "studio_gray": ("kitchen", "gray_floor.png", "gray_wall.png", "gallery"),
    "studio_sky": ("living_room", "marble_floor.png", "canvas_sky_blue.png", "gallery"),
    "library": ("living_room", "dark_floor_texture.png", "stucco_wall.png", "shelves"),
    "workshop": ("kitchen", "gray_ceramic_tile.png", "light_blue_wall.png", "shelves"),
    "loft": ("kitchen", "light_floor.png", "new_light_gray_plaster.png", "none"),
}
for _name, (_base, _floor, _wall, _props) in ROOMS.items():
    ARENAS[_name] = dict(ARENAS[_base], base=_base, floor=_T + _floor, wall=_T + _wall, props=_props)

# Table-top surfaces a visual config may put on its base table (texture file swapped in; geometry unchanged).
TABLE_SURFACES = {
    "default": None,
    "dark_wood": _T + "dark_floor_texture.png",
    "planks": _T + "seamless_wood_planks_floor.png",
    "gray_wood": _T + "dapper_gray_floor.png",
    "rustic": _T + "rustic_floor.png",
    "white_marble": _T + "white_marble_floor.png",
    "gray_tile": _T + "gray_ceramic_tile.png",
    "terracotta": _T + "brown_ceramic_tile.png",
    "slate": _T + "dark_gray_plaster.png",
    "concrete": _T + "stucco_wall.png",
    "laminate_blue": _T + "kona_gotham.png",
    "linen": _T + "yellow_linen_wall_texture.png",
    "green": _T + "dark_green_plaster_wall.png",
}
# Per base geometry: (table-top texture name, table-top material name) in the LIBERO scene.
TABLE_MATERIALS = {"kitchen": ("tex-table", "table_texture"), "living_room": ("tex-living_room_table", "living_room_table")}


def arena_base(name: str) -> str:
    """Table geometry (one of the two validation arenas) a room is built on."""
    return ARENAS[name].get("base", name)


def arenas_with_base(base: str) -> tuple[str, ...]:
    return tuple(name for name in ARENAS if arena_base(name) == base)


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


@functools.lru_cache(maxsize=None)
def _asset_hull_volume(path: str, scale: float) -> float:
    """Cached ``_hull_volume`` of a catalog asset (compiling its visual meshes per build is slow and memory-hungry)."""
    return _hull_volume(load_mjcf(ASSETS / path, scale, "hull"))


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
    mass = spec.mass if spec.mass is not None else _asset_hull_volume(a.path, spec.scale) * a.density
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


def _props(root, prop_set):
    """Copy visual-only backdrop meshes (and their mesh/material/texture assets) from other LIBERO scenes."""
    asset, worldbody = root.find("asset"), root.find("worldbody")
    names = {(el.tag, el.get("name")) for el in asset}
    for i, (xml, mesh, pos, quat) in enumerate(PROP_SETS[prop_set]):
        source = load_mjcf(ASSETS / xml, ARENA_SCALE, prefix=f"prop{i}")
        src_asset = source.find("asset")
        geoms = [g for g in source.find("worldbody") if g.tag == "geom" and g.get("mesh") == f"prop{i}_{mesh}"]
        if not geoms or (pos is not None and len(geoms) != 1):
            raise ValueError(f"prop {mesh} missing or ambiguous in {xml}")
        material = src_asset.find(f"material[@name='{geoms[0].get('material')}']")
        needed = [src_asset.find(f"mesh[@name='{geoms[0].get('mesh')}']"), material]
        if material.get("texture"):
            needed.append(src_asset.find(f"texture[@name='{material.get('texture')}']"))
        for el in needed:
            if (el.tag, el.get("name")) not in names:
                asset.append(copy.deepcopy(el))
                names.add((el.tag, el.get("name")))
        for j, source_geom in enumerate(geoms):
            geom = copy.deepcopy(source_geom)
            geom.attrib.update(contype="0", conaffinity="0", group="1", name=f"backdrop_{i}_{j}")
            for key in ("solimp", "solref", "density", "friction"):
                geom.attrib.pop(key, None)
            if pos is not None:
                _set(geom, "pos", pos)
            if quat is not None:
                _set(geom, "quat", quat)
            worldbody.append(geom)


def _restyle(root, cfg):
    """Swap floor/wall textures and replace the base's backdrop meshes; collision geometry is untouched."""
    for texture in root.find("asset").iter("texture"):
        if texture.get("name") == "texplane":
            texture.set("file", str((ASSETS / cfg["floor"]).resolve()))
        elif texture.get("name") == "tex-wall":
            texture.set("file", str((ASSETS / cfg["wall"]).resolve()))
    worldbody = root.find("worldbody")
    for geom in list(worldbody):
        if geom.tag == "geom" and geom.get("type") == "mesh" and geom.get("contype") == "0":
            worldbody.remove(geom)
    # Drop the removed backdrop's now-unreferenced meshes/materials/textures (they would still be compiled).
    asset = root.find("asset")
    meshes = {g.get("mesh") for g in root.iter("geom")}
    materials = {g.get("material") for g in root.iter("geom")}
    for el in list(asset):
        if (el.tag == "mesh" and el.get("name") not in meshes) or (el.tag == "material" and el.get("name") not in materials):
            asset.remove(el)
    textures = {m.get("texture") for m in asset.iter("material")}
    for el in list(asset):
        if el.tag == "texture" and el.get("type") != "skybox" and el.get("name") not in textures:
            asset.remove(el)
    _props(root, cfg["props"])


def _arena(name):
    cfg = ARENAS[name]
    root = load_mjcf(ASSETS / cfg["xml"], ARENA_SCALE)
    if "base" in cfg:
        _restyle(root, cfg)
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


@functools.lru_cache(maxsize=None)
def table_top_z(arena: str) -> float:
    arena = arena_base(arena)  # rooms share their base table geometry exactly
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


def camera_xml(name, pos, lookat, fovy, roll=None):
    """A fixed camera at `pos` looking at `lookat` with image-up roughly world +z, rolled by `roll` degrees
    (positive turns the image clockwise) about the viewing axis."""
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    fwd = (lookat - pos) / np.linalg.norm(lookat - pos)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    if roll:
        c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        right, up = c * right + s * up, c * up - s * right
    xy = " ".join(f"{v:.6g}" for v in (*right, *up))
    return f'<camera name="{name}" mode="fixed" pos="{" ".join(f"{v:.6g}" for v in pos)}" xyaxes="{xy}" fovy="{fovy}"/>'


def _surfaces(asset_el, arena_name, visual_config):
    """Apply a visual config's table-top surface and table/wall/floor tints (materials only, no geometry)."""
    texture_name, material_name = TABLE_MATERIALS[arena_base(arena_name)]
    surface = getattr(visual_config, "table_surface", None)
    tints = {material_name: getattr(visual_config, "table_tint", None),
             "walls_mat": getattr(visual_config, "wall_tint", None),
             "floorplane": getattr(visual_config, "floor_tint", None)}
    for el in asset_el:
        if el.tag == "texture" and el.get("name") == texture_name and TABLE_SURFACES.get(surface):
            el.set("file", str((ASSETS / TABLE_SURFACES[surface]).resolve()))
        if el.tag == "material" and tints.get(el.get("name")) is not None:
            _set(el, "rgba", [*tints[el.get("name")], 1.0])


def build_scene_xml(spec: SceneSpec, cameras: list[str] = (), *, visual_config=None) -> str:
    """MJCF string for `spec`; `cameras` are extra <camera> elements placed in the worldbody."""
    arena_name = spec.arena if visual_config is None else visual_config.arena
    cfg = ARENAS[arena_name]
    arena = _arena(arena_name)
    mj = ET.Element("mujoco", model=f"so101_val_{arena_name}")
    ET.SubElement(mj, "include", file=str(get_so101_mujoco_model_path()))
    mj.extend(_fragment(MUJOCO_SCENE_OPTION_XML + VISUAL_XML))
    if getattr(visual_config, "ambient", None) is not None:
        _set(mj.find("visual/headlight"), "ambient", [visual_config.ambient] * 3)
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
        _surfaces(arena.find("asset"), arena_name, visual_config)
    asset.extend(arena.find("asset"))
    world.extend(el for el in arena.find("worldbody") if el.tag not in ("camera", "light"))
    worldbody.extend(_fragment(LIGHTS_XML + "".join(cameras)))
    if visual_config is not None:
        for camera in list(worldbody.findall("camera")):
            if camera.get("name") == "front":
                worldbody.remove(camera)
        front = visual_config.front_camera
        worldbody.extend(_fragment(camera_xml("front", front.pos, front.lookat, front.fovy,
                                              getattr(front, "roll", None))))
        for sampled in visual_config.lights:
            light = worldbody.find(f"light[@name='{sampled.name}']")
            if light is None:  # optional extra spot light (e.g. "rim")
                light = ET.SubElement(worldbody, "light", name=sampled.name, directional="false", cutoff="50",
                                      exponent="1", specular="0.1 0.1 0.1", castshadow="false", bulbradius="0.06")
            _set(light, "pos", sampled.pos)
            _set(light, "dir", np.array([0.18, 0.0, 0.0]) - sampled.pos)
            _set(light, "diffuse", np.array(sampled.color) * sampled.intensity)
            if getattr(sampled, "castshadow", None) is not None:
                light.set("castshadow", str(bool(sampled.castshadow)).lower())
    for spec_obj in spec.free_bodies:
        _free_body(mj, asset, worldbody, spec_obj)
    for fx in spec.fixtures:
        _fixture(mj, asset, worldbody, fx)
    if visual_config is not None:
        _limit_textures(mj)
    return ET.tostring(mj, encoding="unicode")


TEXTURE_LIMIT = 1024  # px: the 640x480 cameras never resolve more on a tabletop object
TEXTURE_CACHE = Path.home() / ".cache" / "so101_sim" / "textures"


def _limit_textures(mj):
    """Training scenes only: point every texture larger than TEXTURE_LIMIT at a downscaled cached copy.

    LIBERO bowls/plate and the GSO/YCB scans ship 4096^2 textures (48 MB each once decoded), so a scene with
    distractors held ~0.5 GB of texture data per environment. Downscaling changes no geometry or physics."""
    for el in mj.iter("texture"):
        if "file" in el.attrib:
            el.set("file", str(_downscaled(Path(el.get("file")))))


@functools.lru_cache(maxsize=None)
def _downscaled(path: Path) -> Path:
    import hashlib
    import os
    from PIL import Image
    with Image.open(path) as image:
        if max(image.size) <= TEXTURE_LIMIT:
            return path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out = TEXTURE_CACHE / f"{path.stem}_{digest}_{TEXTURE_LIMIT}.png"
        if not out.exists():
            factor = TEXTURE_LIMIT / max(image.size)
            size = tuple(max(1, round(x * factor)) for x in image.size)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".{os.getpid()}.tmp.png")
            image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB").resize(size, Image.LANCZOS).save(tmp)
            os.replace(tmp, out)
        return out
