import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scipy.spatial import ConvexHull
from so101_nexus import get_so101_mujoco_model_path
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML, SCENE_LIGHTS_XML, SCENE_VISUAL_XML

ASSETS = Path(__file__).resolve().parents[1] / "third_party/LIBERO/libero/libero/assets"
ARENA = ASSETS / "scenes/libero_living_room_tabletop_base_style.xml"
OBJECTS = {
    "basket": ASSETS / "stable_scanned_objects/basket/basket.xml",
    "alphabet_soup": ASSETS / "stable_hope_objects/alphabet_soup/alphabet_soup.xml",
    "cream_cheese": ASSETS / "stable_hope_objects/cream_cheese/cream_cheese.xml",
    "tomato_sauce": ASSETS / "stable_hope_objects/tomato_sauce/tomato_sauce.xml",
    "ketchup": ASSETS / "stable_hope_objects/ketchup/ketchup.xml",
}
UPRIGHT = {
    "basket": [(0, 0)],
    "alphabet_soup": [(0, np.pi / 2)],
    "cream_cheese": [],
    "tomato_sauce": [(0, np.pi / 2)],
    "ketchup": [(0, np.pi / 2), (2, np.pi / 2)],
}
EFFECTIVE_DENSITY = {
    "basket": 150,
    "alphabet_soup": 900,
    "cream_cheese": 900,
    "tomato_sauce": 900,
    "ketchup": 700,
}
REFS = ("mesh", "material", "texture")


def _vec(el, key):
    return np.array(el.get(key).split(), dtype=float)


def _set(el, key, v):
    el.set(key, " ".join(f"{x:.6g}" for x in np.atleast_1d(v)))


def _absolute_files(root, base):
    for el in root.iter():
        if "file" in el.attrib and not el.get("file").startswith("/"):
            el.set("file", str((base / el.get("file")).resolve()))


def _scale(root, s):
    for el in root.iter():
        if el.tag in ("body", "geom", "site", "camera", "light", "inertial") and "pos" in el.attrib:
            _set(el, "pos", _vec(el, "pos") * s)
        if el.tag in ("geom", "site") and "size" in el.attrib and el.get("type") != "mesh":
            _set(el, "size", _vec(el, "size") * s)
        if el.tag == "mesh":
            _set(el, "scale", (_vec(el, "scale") if "scale" in el.attrib else np.ones(3)) * s)
        if el.tag == "inertial" and "mass" in el.attrib:
            _set(el, "mass", float(el.get("mass")) * s**3)


def _prefix(root, p):
    for el in root.iter():
        for key in ("name",) + REFS:
            if key in el.attrib:
                el.set(key, f"{p}_{el.get(key)}")


def _hide_collision(root):
    for el in root.iter("geom"):
        el.attrib.pop("solref", None)
        el.attrib.pop("solimp", None)
        if el.get("group") == "0":
            el.set("group", "3")


def _upright_quat(steps):
    q = np.array([1.0, 0, 0, 0])
    for axis, angle in steps:
        r = np.zeros(4)
        mujoco.mju_axisAngle2Quat(r, np.eye(3)[axis], angle)
        mujoco.mju_mulQuat(q, r, q.copy())
    return q


def _load(path, s, prefix=None):
    root = ET.parse(path).getroot()
    _absolute_files(root, path.parent)
    _scale(root, s)
    _hide_collision(root)
    if prefix:
        _prefix(root, prefix)
    return root


def _hull_volume(root):
    mj = ET.Element("mujoco")
    mj.append(copy.deepcopy(root.find("asset")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    return ConvexHull(model.mesh_vert[: model.mesh_vertnum[0]]).volume


def _set_object_physics(root, mass):
    boxes = [el for el in root.iter("geom") if el.get("contype") != "0"]
    volume = sum(8 * np.prod(_vec(el, "size")) for el in boxes)
    for el in root.iter("geom"):
        if el.get("contype") == "0":
            el.set("density", "0")
    for el in boxes:
        el.attrib.update(density=f"{mass / volume:.6g}", condim="4", friction="1 0.05 0.001")


def _xml_fragment(xml):
    return list(ET.fromstring(f"<root>{xml}</root>"))


def build_scene_xml(scale, robot_pos, table_top_z):
    arena = _load(ARENA, scale)
    mj = ET.Element("mujoco", model="libero_living_room_so101")
    ET.SubElement(mj, "include", file=str(get_so101_mujoco_model_path()))
    mj.extend(_xml_fragment(MUJOCO_SCENE_OPTION_XML + SCENE_VISUAL_XML))
    asset = ET.SubElement(mj, "asset")
    worldbody = ET.SubElement(mj, "worldbody")
    offset = np.array([-robot_pos[0], -robot_pos[1], -table_top_z])
    world = ET.SubElement(worldbody, "body", name="libero_world")
    _set(world, "pos", offset)
    asset.extend(arena.find("asset"))
    for el in arena.find("worldbody"):
        if el.tag != "camera":
            world.append(el)
    worldbody.extend(_xml_fragment(SCENE_LIGHTS_XML))
    for name, path in OBJECTS.items():
        obj = _load(path, scale, name)
        _set_object_physics(obj, _hull_volume(obj) * EFFECTIVE_DENSITY[name])
        asset.extend(obj.find("asset"))
        inner = obj.find("worldbody/body")
        body = ET.SubElement(worldbody, "body", name=name, pos="0 0 -1")
        ET.SubElement(body, "joint", name=f"{name}_joint", type="free", damping="0.0005")
        upright = ET.SubElement(body, "body", name=f"{name}_upright")
        _set(upright, "quat", _upright_quat(UPRIGHT[name]))
        for child in inner.find("body"):
            upright.append(child)
        for child in inner:
            if child.tag == "site":
                upright.append(child)
    return ET.tostring(mj, encoding="unicode")


def table_top_z(scale, x=0.0, y=0.0):
    arena = _load(ARENA, scale)
    mj = ET.Element("mujoco")
    mj.append(arena.find("asset"))
    wb = ET.SubElement(mj, "worldbody")
    wb.extend(copy.deepcopy(list(arena.find("worldbody"))))
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geomid = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0, 0, -1.0]), None, 1, -1, geomid)
    return 5.0 - d
