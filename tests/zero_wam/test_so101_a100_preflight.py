"""Unit tests for the pure helpers of zero_wam.so101_a100_preflight.

No GPU, upstream Zero-WAM import, or dataset fixtures required: GPU query and
topology outputs are synthetic strings; masks are plain numpy arrays.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.so101_a100_preflight as pf


def _query_csv(names=("NVIDIA A100-SXM4-80GB",) * 8, indices=range(8)):
    header = (
        "index, name, memory.total [MiB], compute_cap, "
        "driver_version, pci.bus_id"
    )
    lines = [
        f"{i}, {name}, 81920 MiB, 8.0, 570.133.20, "
        f"00000000:{0x10 + i:02X}:00.0"
        for i, name in zip(indices, names)
    ]
    return header + "\n" + "\n".join(lines) + "\n"


def _topo(link="NV12", n=8):
    cols = [f"GPU{i}" for i in range(n)]
    lines = ["\t" + "\t".join(cols + ["CPU", "Affinity"])]
    for i in range(n):
        cells = [link if j != i else "X" for j in range(n)]
        lines.append(f"GPU{i}\t" + "\t".join(cells + ["0-95", "0"]))
    return "\n".join(lines) + "\n"


# ---- GPU query parsing / contract ------------------------------------------


def test_parse_gpu_query_csv_with_header():
    rows = pf.parse_gpu_query_csv(_query_csv())
    assert len(rows) == 8
    assert rows[0]["index"] == 0
    assert rows[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert rows[0]["memory_mib"] == 81920
    assert rows[3]["pci_bus_id"].startswith("00000000:")


def test_parse_gpu_query_csv_no_header():
    rows = pf.parse_gpu_query_csv(
        "\n".join(_query_csv().splitlines()[1:])
    )
    assert len(rows) == 8


def test_check_gpu_rows_good():
    rows = pf.parse_gpu_query_csv(_query_csv())
    assert pf.check_gpu_rows(rows) == []


def test_check_gpu_rows_seven_gpus():
    rows = pf.parse_gpu_query_csv(_query_csv(indices=range(7)))
    problems = pf.check_gpu_rows(rows)
    assert any("8" in p and "7" in p for p in problems)


def test_check_gpu_rows_wrong_model():
    names = ("NVIDIA A100-SXM4-80GB",) * 7 + ("NVIDIA H100 80GB HBM3",)
    rows = pf.parse_gpu_query_csv(_query_csv(names=names))
    problems = pf.check_gpu_rows(rows)
    assert any("H100" in p for p in problems)


def test_check_gpu_rows_low_memory():
    header, *lines = _query_csv().splitlines()
    lines[2] = lines[2].replace("81920 MiB", "40960 MiB")
    rows = pf.parse_gpu_query_csv(header + "\n" + "\n".join(lines))
    problems = pf.check_gpu_rows(rows)
    assert any("memory" in p and "GPU 2" in p for p in problems)


def test_check_gpu_rows_duplicate_index():
    rows = pf.parse_gpu_query_csv(
        _query_csv(indices=[0, 1, 2, 3, 4, 5, 6, 6])
    )
    assert pf.check_gpu_rows(rows)


# ---- NVLink topology --------------------------------------------------------


def test_check_nv12_topology_good():
    assert pf.check_nv12_topology(_topo()) == []


def test_check_nv12_topology_missing_nv12():
    lines = _topo().splitlines()
    # replace GPU2->GPU5 link with PIX (header + 6th cell of row GPU2)
    row = lines[1 + 2].split("\t")
    row[1 + 5] = "PIX"
    lines[1 + 2] = "\t".join(row)
    problems = pf.check_nv12_topology("\n".join(lines))
    assert any("GPU2->GPU5" in p for p in problems)


def test_check_nv12_topology_unparsed():
    assert pf.check_nv12_topology("garbage\nno gpus\n")


# ---- atomic JSON writer -----------------------------------------------------


def test_atomic_write_json_creates_parents_and_file(tmp_path):
    out = tmp_path / "nested" / "deep" / "report.json"
    pf.atomic_write_json(out, {"ok": True, "n": 3})
    assert json.loads(out.read_text()) == {"ok": True, "n": 3}
    leftovers = list(out.parent.glob(".*.tmp-*"))
    assert leftovers == []


def test_atomic_write_json_overwrites(tmp_path):
    out = tmp_path / "r.json"
    pf.atomic_write_json(out, {"v": 1})
    pf.atomic_write_json(out, {"v": 2})
    assert json.loads(out.read_text())["v"] == 2


# ---- action-mask active channels --------------------------------------------


def test_active_mask_channels_expected():
    mask = np.zeros((30, 4, 8, 1), dtype=bool)
    mask[[0, 1, 2, 3, 4, 28]] = True
    assert pf.active_mask_channels(mask) == [0, 1, 2, 3, 4, 28]


def test_active_mask_channels_empty_and_partial():
    mask = np.zeros((30, 4, 8, 1), dtype=bool)
    assert pf.active_mask_channels(mask) == []
    mask[7, 0, 0, 0] = True
    mask[28, :, :, 0] = True
    assert pf.active_mask_channels(mask) == [7, 28]


def test_module_import_has_no_torch_or_upstream_dependency():
    # top-level import of the module must not pull torch or upstream wan_va
    assert "torch" not in vars(pf)
    assert "wan_va" not in vars(pf)
    assert "easydict" not in vars(pf)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
