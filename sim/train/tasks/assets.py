"""Object catalog for training tasks: LIBERO assets, primitives and so101-nexus scanned objects.

* ``Obj(name, catalog_name, scale=0.5)`` -- LIBERO objects in ``sim.val.scene.CATALOG`` (see INVENTORY).
* ``Block``/``mat`` primitives -- boxes of any size/colour (``sim.val.scene.Block``, ``base.mat``).
* ``Scanned(name, "gso"|"ycb", model_id)`` -- Google Scanned Objects / YCB meshes from so101-nexus,
  downloaded on first use from the public Hugging Face mirrors (no paid API). Collision is the nexus
  convex decomposition, or a single convex hull when the optional ``coacd`` package is absent (it is
  absent in this venv), so concave scans (tape-roll hole, mug interior) collide as filled hulls.

``INVENTORY`` lists measured sizes (cm, at the given scale) and masses (g); "graspable" means the
narrowest horizontal side fits the SO-101 jaws (fingertip gap 68.6 mm at OPEN; <= 5 cm leaves room
for the 3-12 mm approach margins). Regenerate with ``python -m sim.train.tasks.assets``.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass
class Scanned:
    """A free scanned-mesh object (GSO or YCB via so101-nexus), resting in its nexus-validated pose."""

    name: str
    source: str            # "gso" or "ycb"
    model_id: str
    scale: float = 1.0     # uniform mesh scale (nexus meshes are real-world size; 0.5 matches the LIBERO convention)
    mass: float | None = None  # default: the nexus mass times scale**3
    friction: float = 1.0
    rgba: tuple | None = None

    def _accessors(self):
        from so101_nexus.object_slots import _accessors_for
        from so101_nexus.objects import GSOObject, YCBObject
        obj = {"gso": GSOObject, "ycb": YCBObject}[self.source](self.model_id)
        return obj, _accessors_for(obj)

    def build_mjcf(self, mj, asset_el, worldbody):
        from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose
        obj, acc = self._accessors()
        acc.ensure_assets(self.model_id)
        parts = acc.collision_parts(self.model_id)
        mass = self.mass if self.mass is not None else acc.default_mass(self.model_id) * self.scale ** 3
        prefix = f"{self.name}_scan"
        coll = [(f"{prefix}_coll_{k}", part) for k, part in enumerate(parts)]
        scale = " ".join([f"{self.scale:.6g}"] * 3)
        for mesh, part in coll:
            ET.SubElement(asset_el, "mesh", name=mesh, file=part.path.as_posix(), scale=scale)
        ET.SubElement(asset_el, "mesh", name=f"{prefix}_vis", file=acc.visual_mesh(self.model_id).as_posix(),
                      scale=scale)
        material = None
        texture = acc.texture_file(self.model_id)
        if texture.exists():
            ET.SubElement(asset_el, "texture", name=f"{prefix}_tex", type="2d", file=texture.as_posix())
            material = f"{prefix}_mat"
            attrs = dict(name=material, texture=f"{prefix}_tex", texuniform="false", specular="0.1", reflectance="0")
            if self.rgba is not None:
                attrs["rgba"] = " ".join(map(str, self.rgba))
            ET.SubElement(asset_el, "material", **attrs)
        quat = self._rest(coll)[0]
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat=" ".join(f"{q:.6g}" for q in quat))
        for (mesh, part) in coll:
            ET.SubElement(upright, "geom", name=mesh, type="mesh", mesh=mesh, mass=repr(float(mass) * part.mass_fraction),
                          group="3", condim="4", friction=f"{self.friction} 0.02 0.001")
        vis = dict(name=f"{prefix}_visual", type="mesh", mesh=f"{prefix}_vis", group="1", contype="0",
                   conaffinity="0", mass="0")
        if material:
            vis["material"] = material
        ET.SubElement(upright, "geom", **vis)

    def _rest(self, coll):
        """(rest quaternion, rest-frame AABB low, high) of the collision parts; cached per model and scale."""
        return _rest_geometry(self.source, self.model_id, float(self.scale), tuple((m, str(p.path)) for m, p in coll))

    @staticmethod
    def _body_verts(coll, scale=1.0):
        mj = ET.Element("mujoco")
        asset = ET.SubElement(mj, "asset")
        body = ET.SubElement(ET.SubElement(mj, "worldbody"), "body", name="b")
        ET.SubElement(body, "freejoint")
        for mesh, part in coll:
            ET.SubElement(asset, "mesh", name=mesh, file=part.path.as_posix(), scale=" ".join([f"{scale:.6g}"] * 3))
            ET.SubElement(body, "geom", type="mesh", mesh=mesh, mass="0.1")
        m = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
        verts = []
        for g in range(m.ngeom):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            verts.append(np.einsum("ij,nj->ni", rot.reshape(3, 3), v.astype(np.float64)) + m.geom_pos[g])
        verts = np.vstack(verts)
        if not np.isfinite(verts).all():
            raise ValueError("non-finite scanned collision vertices")
        return verts


@dataclass
class ScannedBox(Scanned):
    """A scanned object whose collision is one box fitted to its rest-pose extents (visual: the scan).

    For distractors that are never grasped: convex-hull scans rock and creep millimetres per second on the
    table (measured 2026-10-07: gelatin box 17 mm, sponge 11 mm in 6 s), while a box settles exactly like
    the LIBERO grocery boxes. The mesh rests on the box bottom, so it looks as if it lies on the table."""

    def build_mjcf(self, mj, asset_el, worldbody):
        from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose
        obj, acc = self._accessors()
        acc.ensure_assets(self.model_id)
        parts = acc.collision_parts(self.model_id)
        mass = self.mass if self.mass is not None else acc.default_mass(self.model_id) * self.scale ** 3
        prefix = f"{self.name}_scan"
        coll = [(f"{prefix}_coll_{k}", part) for k, part in enumerate(parts)]
        quat, lo, hi = self._rest(coll)
        scale = " ".join([f"{self.scale:.6g}"] * 3)
        ET.SubElement(asset_el, "mesh", name=f"{prefix}_vis", file=acc.visual_mesh(self.model_id).as_posix(),
                      scale=scale)
        material = None
        texture = acc.texture_file(self.model_id)
        if texture.exists():
            ET.SubElement(asset_el, "texture", name=f"{prefix}_tex", type="2d", file=texture.as_posix())
            material = f"{prefix}_mat"
            attrs = dict(name=material, texture=f"{prefix}_tex", texuniform="false", specular="0.1", reflectance="0")
            if self.rgba is not None:
                attrs["rgba"] = " ".join(map(str, self.rgba))
            ET.SubElement(asset_el, "material", **attrs)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "geom", name=f"{prefix}_box", type="box", group="3", condim="4",
                      pos=" ".join(f"{x:.6g}" for x in (lo + hi) / 2),
                      size=" ".join(f"{x:.6g}" for x in (hi - lo) / 2), mass=repr(float(mass)),
                      friction=f"{self.friction} 0.02 0.001")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat=" ".join(f"{q:.6g}" for q in quat))
        vis = dict(name=f"{prefix}_visual", type="mesh", mesh=f"{prefix}_vis", group="1", contype="0",
                   conaffinity="0", mass="0")
        if material:
            vis["material"] = material
        ET.SubElement(upright, "geom", **vis)


@functools.lru_cache(maxsize=None)
def _rest_geometry(source, model_id, scale, parts):
    """Rest pose and rest-frame extents of a scan's collision parts (pure function of the files; cached so an
    environment build does not recompile every scan's hull)."""
    from types import SimpleNamespace
    from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose
    coll = [(mesh, SimpleNamespace(path=Path(path))) for mesh, path in parts]
    verts = Scanned._body_verts(coll, scale)
    quat = np.asarray(get_mujoco_ycb_rest_pose(verts, model_id=model_id)[0], dtype=np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rested = np.einsum("ij,nj->ni", rot.reshape(3, 3), verts)
    return tuple(float(q) for q in quat), rested.min(axis=0), rested.max(axis=0)


GSO_IDS = ("Pony_C_Clamp_1440", "Cole_Hardware_Mini_Honey_Dipper", "OXO_Soft_Works_Can_Opener_SnapLock",
           "3M_Vinyl_Tape_Green_1_x_36_yd", "Shurtape_Gaffers_Tape_Silver_2_x_60_yd",
           "Big_O_Sponges_Assorted_Cellulose_12_pack", "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup", "CoQ10",
           "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold",
           "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle",
           "Black_Elderberry_Syrup_54_oz_Gaia_Herbs", "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g")
YCB_IDS = ("009_gelatin_box", "011_banana", "030_fork", "031_spoon", "032_knife", "033_spatula", "037_scissors",
           "040_large_marker", "043_phillips_screwdriver", "058_golf_ball")

# Measured by ``python -m sim.train.tasks.assets > sim/train/tasks/asset_inventory.json`` (2026-10-06):
# dims are settled axis-aligned collision extents (x, y, z) in cm (conservative for rotated/elongated meshes);
# "upright" = within 10 deg of its rest pose after 2 s on the table; key suffix @scale for scanned objects.
# YCB masses are the nexus flat default (10 g at 1.0x): pass a realistic ``mass`` when using them.
INVENTORY: dict = json.loads((Path(__file__).with_name("asset_inventory.json")).read_text())


def measure(specs, settle_seconds=2.0):
    """Build a scene of the given free-body specs, settle each on the table, report size/mass/stability."""
    from sim.val.env import ValEnv
    from sim.val.scene import SceneSpec

    class Probe(ValEnv):
        def make_scene(self):
            return SceneSpec(arena="kitchen", objects=list(specs))

        def layout(self):
            for i, spec in enumerate(specs):
                self.set_object_pose(spec.name, (20 + i, 3.0), z=self._floor_z)

        def success(self):
            return False

    env = Probe(render_images=False)
    env.reset(seed=0)
    out = {}
    for spec in specs:
        name = spec.name
        env.set_object_pose(name, (0.22, 0.0))
        mujoco.mj_forward(env.model, env.data)
        start = env.data.xquat[env._body[name]].copy()
        for _ in range(int(settle_seconds * 210)):
            mujoco.mj_step(env.model, env.data)
        q = env.data.xquat[env._body[name]]
        tilt = np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(q, start))), 0, 1)))
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in env._geoms[name]:
            c = env.data.geom_xpos[g] + env.data.geom_xmat[g].reshape(3, 3) @ env.model.geom_aabb[g, :3]
            half = np.abs(env.data.geom_xmat[g].reshape(3, 3)) @ env.model.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
        dims = (hi - lo) * 100
        out[name] = dict(dims_cm=[round(float(x), 1) for x in dims],
                         mass_g=round(float(env.model.body_subtreemass[env._body[name]]) * 1000, 1),
                         upright=bool(tilt < 10), graspable=bool(min(dims[:2]) <= 5.0))
        env.set_object_pose(name, (20 + len(out), 3.0), z=env._floor_z)
    env.close()
    return out


if __name__ == "__main__":
    from sim.val.scene import CATALOG, Obj
    libero = [Obj(n, n) for n in CATALOG if n not in ("microwave", "flat_stove")]
    scanned = [Scanned(f"{src}_{i}_{k}", src, m, scale=scale) for scale, k in ((1.0, "full"), (0.5, "half"))
               for src, ids in (("gso", GSO_IDS), ("ycb", YCB_IDS)) for i, m in enumerate(ids)]
    result = {}
    for spec, value in measure(libero).items():
        result[f"libero:{spec}"] = value
    for spec in scanned:
        try:
            result[f"{spec.source}:{spec.model_id}@{spec.scale}"] = measure([spec])[spec.name]
        except Exception as exc:  # report, never hide, an asset that fails to load
            result[f"{spec.source}:{spec.model_id}@{spec.scale}"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    print(json.dumps(result, indent=1))


# ----- extra distractors ---------------------------------------------------------------------------------------
# Low (<= 3.1 cm), never-grasped clutter added per episode on top of each family's distractor pool (selected by
# the episode's visual configuration). Measured 2026-10-07 with 5 yaws on both table geometries: after the reset
# settle, 0.00 mm drift, 0 rad rotation and no penetration beyond the contact margin over 6 s (scans use
# ``ScannedBox``: hull-collision scans crept up to 17 mm). Footprint radius <= 7.6 cm.
# name -> (spec factory, guard words). An extra is never used when a guard word appears in the task's
# instruction, steps, object names/kinds or goal, so it cannot duplicate a task object's kind or make the
# instruction ambiguous.
def _libero(name, asset, rgba=None):
    from sim.val.scene import Obj
    return lambda: Obj(name, asset, rgba=rgba)


def _scan(name, source, model_id, mass):
    return lambda: ScannedBox(name, source, model_id, scale=0.5, mass=mass)


# name -> (spec factory, guard words, settled height cm, LOW_DISTRACTORS look-alike or None).
EXTRA_DISTRACTORS = {
    "cookies": (_libero("cookies", "cookies"), ("cookie", "box", "package"), 0.9, "cream_cheese"),
    "macaroni_and_cheese": (_libero("macaroni_and_cheese", "macaroni_and_cheese"), ("macaroni", "cheese", "box"),
                            1.3, "cream_cheese"),
    "salad_dressing": (_libero("salad_dressing", "salad_dressing"), ("dressing", "bottle", "sauce"), 1.8, None),
    "new_salad_dressing": (_libero("new_salad_dressing", "new_salad_dressing"), ("dressing", "bottle", "sauce"),
                           1.8, None),
    "black_book_flat": (_libero("black_book_flat", "black_book_flat"), ("book",), 1.4, None),
    "yellow_book_flat": (_libero("yellow_book_flat", "yellow_book_flat"), ("book",), 1.2, None),
    "blue_plate": (_libero("blue_plate", "plate", (0.45, 0.6, 1.0, 1.0)), ("plate", "dish"), 0.9, "plate"),
    "green_bowl": (_libero("green_bowl", "white_bowl", (0.5, 0.9, 0.55, 1.0)), ("bowl",), 1.7, "white_bowl"),
    "yellow_ramekin": (_libero("yellow_ramekin", "ramekin", (1.0, 0.85, 0.4, 1.0)), ("ramekin", "bowl", "cup"),
                       2.1, "ramekin"),
    "coq10_bottle": (_scan("coq10_bottle", "gso", "CoQ10", 0.03), ("bottle",), 2.4, None),
    "sprinkles": (_scan("sprinkles", "gso", "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold", 0.03),
                  ("sprinkle", "canister", "can"), 2.3, "alphabet_soup"),
    "raisinets": (_scan("raisinets", "gso", "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g", 0.03),
                  ("raisinet", "chocolate", "box", "package"), 1.1, "cream_cheese"),
    "gelatin_box": (_scan("gelatin_box", "ycb", "009_gelatin_box", 0.025), ("gelatin", "box", "jello"), 1.5,
                    "cream_cheese"),
    "banana": (_scan("banana", "ycb", "011_banana", 0.03), ("banana", "fruit"), 1.8, None),
    "vinyl_tape": (_scan("vinyl_tape", "gso", "3M_Vinyl_Tape_Green_1_x_36_yd", 0.02), ("tape", "roll"), 1.3, None),
    "sponge_pad": (_scan("sponge_pad", "gso", "Big_O_Sponges_Assorted_Cellulose_12_pack", 0.01), ("sponge",), 0.9,
                   None),
    "glazed_cup": (_scan("glazed_cup", "gso", "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup", 0.03),
                   ("cup", "ramekin", "bowl"), 2.2, "ramekin"),
    "fork": (_scan("fork", "ycb", "030_fork", 0.02), ("fork", "utensil", "cutlery"), 0.8, None),
    "spoon": (_scan("spoon", "ycb", "031_spoon", 0.02), ("spoon", "utensil", "cutlery"), 1.0, None),
    "marker": (_scan("marker", "ycb", "040_large_marker", 0.01), ("marker", "pen"), 0.9, None),
    "screwdriver": (_scan("screwdriver", "ycb", "043_phillips_screwdriver", 0.02), ("screwdriver", "tool"), 1.8,
                    None),
}


def eligible_extra_distractors(texts, pool) -> tuple[str, ...]:
    """Extras allowed beside a family distractor ``pool`` for a task described by ``texts``.

    Dropped when a guard word appears in the instruction, steps, object names/kinds or goal; when the family
    pool deliberately leaves out the extra's LOW_DISTRACTORS look-alike (e.g. no plate where a plate reads as a
    round target); or when the extra is taller than every item of the family pool (short-only pools).
    """
    import re
    from .base import LOW_DISTRACTORS
    words = set(re.findall(r"[a-z0-9]+", " ".join(str(t) for t in texts).lower().replace("_", " ")))
    words |= {w[:-1] for w in words if w.endswith("s") and len(w) > 3}
    heights = [INVENTORY[f"libero:{n}"]["dims_cm"][2] for n in pool if f"libero:{n}" in INVENTORY]
    tallest = max(heights, default=0.0)
    out = []
    for name, (_, guards, height, analog) in EXTRA_DISTRACTORS.items():
        if any(g in words for g in guards) or name in words or height > tallest + 1e-9:
            continue
        if analog is not None and analog in LOW_DISTRACTORS and analog not in pool:
            continue
        out.append(name)
    return tuple(out)
