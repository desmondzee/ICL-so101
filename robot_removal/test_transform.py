"""Assert-based tests for the Zero-WAM transform, composite, and mask utils.
Run: /venv/main/bin/python -m robot_removal.test_transform
"""

import math

import numpy as np
from PIL import Image

from .masks import disk_dilate, select_robot
from .postprocess import composite, zerowam_transform


def check(name, cond):
    assert cond, name
    print(f"  ok {name}")


def t_transform():
    # 640x480 -> 480x360 cover -> crop (0,20,480,340)
    im = Image.new("RGB", (640, 480), (10, 20, 30))
    out, meta = zerowam_transform(im)
    check("640x480 size", out.size == (480, 320))
    check("640x480 crop", meta["crop"] == [0, 20, 480, 340])

    # wider than target ratio: 960x480 (2:1 vs 1.5:1)
    im = Image.new("RGB", (960, 480))
    out, meta = zerowam_transform(im)
    check("960x480 size", out.size == (480, 320))
    check("960x480 cover-height", meta["resized"] == [640, 320])
    check("960x480 side crop", meta["crop"] == [80, 0, 560, 320])

    # taller: 480x960 -> cover width -> 480x960*... scale=max(1, .33)=1 -> wait scale=480/480=1 vs 320/960=.33 -> 1 -> 480x960, crop vertically
    im = Image.new("RGB", (480, 960))
    out, meta = zerowam_transform(im)
    check("480x960 size", out.size == (480, 320))
    check("480x960 top crop", meta["crop"] == [0, 320, 480, 640])

    # exact size passthrough
    im = Image.new("RGB", (480, 320))
    out, meta = zerowam_transform(im)
    check("480x320 passthrough size", out.size == (480, 320))
    check("480x320 passthrough crop", meta["crop"] == [0, 0, 480, 320])

    # odd dims: 641x479 -> scale=0.7511... ceil->481? ceil(641*.7512)=482, ceil(479*.6681)=320
    w, h = 641, 479
    im = Image.new("RGB", (w, h))
    out, meta = zerowam_transform(im)
    s = max(480 / w, 320 / h)
    check("odd cover", meta["resized"] == [math.ceil(w * s), math.ceil(h * s)])
    check("odd size", out.size == (480, 320))

    # landmark preservation: a colored pixel at center stays centered
    im = Image.new("RGB", (640, 480), (0, 0, 0))
    im.putpixel((320, 240), (255, 0, 0))
    out, _ = zerowam_transform(im)
    px = np.asarray(out)
    ys, xs = np.where(px[:, :, 0] > 100)
    check("landmark survives", len(ys) > 0)
    check("landmark centered x", abs(xs.mean() - 240) < 3)


def t_composite():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    m = np.zeros((100, 120), bool)
    m[30:60, 40:80] = True
    out = composite(a, b, m)
    check("inside=edit", (out[m] == b[m]).all())
    check("outside=orig", (out[~m] == a[~m]).all())
    check("no aliasing", out.dtype == np.uint8)


def t_masks():
    # disk dilation radius
    m = np.zeros((101, 101), bool)
    m[50, 50] = True
    d = disk_dilate(m, 8)
    ys, xs = np.where(d)
    check("disk radius", xs.max() - 50 == 8 and ys.max() - 50 == 8)
    check("disk corner cut", not d[50 - 7, 50 - 7])  # sqrt(98)>8 -> outside

    # selection: best arm inside box, extras touching only
    arm1 = np.zeros((100, 100), bool); arm1[20:40, 30:50] = True   # inside box
    arm2 = np.zeros((100, 100), bool); arm2[60:80, 70:90] = True   # outside box
    ext = np.zeros((100, 100), bool); ext[15:25, 30:50] = True     # touches arm1
    ext_far = np.zeros((100, 100), bool); ext_far[85:95, 0:10] = True
    box = (25, 15, 35, 35)  # x,y,w,h covering arm1
    sel = select_robot([arm2, arm1], [ext, ext_far], box)
    check("arm selected", (sel & arm1).any())
    check("other arm dropped", not (sel & arm2).any())
    check("touching extra kept", (sel & ext).any())
    check("far extra dropped", not (sel & ext_far).any())

    # oversized extra rejected
    big = np.zeros((100, 100), bool); big[0:5, :] = True  # 500px vs arm ~400 -> touches? no; ensure non-touch anyway
    sel2 = select_robot([arm1], [big], box)
    check("big extra dropped", not (sel2 & big).any())


if __name__ == "__main__":
    t_transform()
    t_composite()
    t_masks()
    print("all tests passed")
