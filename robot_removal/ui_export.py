"""Export the API-format workflow as ComfyUI UI-format JSON (nodes/links).

Slot indices come from the node's declared input/output order in the live
/object_info so the exported graph is consistent with this ComfyUI version.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .workflow import build_edit_prompt, build_seg_prompt


def _node_info(base: str, class_type: str) -> dict:
    d = json.load(urllib.request.urlopen(f"{base}/object_info/{class_type}"))[class_type]
    return d


def to_ui(prompt: dict, base: str = "http://127.0.0.1:18188") -> dict:
    """Convert API prompt {id: {class_type, inputs}} to UI format."""
    infos = {n["class_type"]: _node_info(base, n["class_type"]) for n in prompt.values()}
    linkable = {}
    for nid, node in prompt.items():
        info = infos[node["class_type"]]
        names = []
        for sec in ("required", "optional"):
            for name, spec in info["input"].get(sec, {}).items():
                # link slots are non-primitive types or forced inputs
                if isinstance(spec[0], str) and spec[0] not in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") or spec[0] == "COMFY_AUTOGROW_V3":
                    names.append(name)
        linkable[nid] = names

    nodes, links = [], []
    lid = 0
    for i, (nid, node) in enumerate(prompt.items()):
        ct = node["class_type"]
        info = infos[ct]
        declared = set(linkable[nid]) | {k for sec in ("required", "optional") for k in info["input"].get(sec, {})}
        inputs, widgets = [], []
        order = info.get("input_order", {})
        for sec in ("required", "optional"):
            for name in order.get(sec, []):
                spec = info["input"].get(sec, {}).get(name)
                if spec is None:
                    continue
                if name in linkable[nid] and spec[0] != "COMFY_AUTOGROW_V3":
                    val = node["inputs"].get(name)
                    entry = {"name": name, "type": spec[0], "link": None}
                    inputs.append(entry)
                    if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                        lid += 1
                        entry["link"] = lid
                        links.append([lid, int(val[0]), int(val[1]), int(nid), len(inputs) - 1, spec[0] if isinstance(spec[0], str) else "IMAGE"])
                elif spec[0] != "COMFY_AUTOGROW_V3" and name in node["inputs"]:
                    widgets.append(node["inputs"][name])
        # autogrow members arrive as dotted keys ("images.image_1")
        for k, v in node["inputs"].items():
            if "." in k and k.split(".", 1)[0] in declared and isinstance(v, list):
                lid += 1
                inputs.append({"name": k, "type": "IMAGE", "link": lid})
                links.append([lid, int(v[0]), int(v[1]), int(nid), len(inputs) - 1, "IMAGE"])
        outputs = [{"name": o, "type": o.upper(), "links": []} for o in info.get("output_name", info.get("output", []))]
        nodes.append(
            {
                "id": int(nid),
                "type": ct,
                "pos": [40 + (i % 5) * 380, 40 + (i // 5) * 420],
                "size": [340, 200],
                "flags": {},
                "order": i,
                "mode": 0,
                "inputs": inputs,
                "outputs": outputs,
                "properties": {"Node name for S&R": ct},
                "widgets_values": widgets,
            }
        )
    # fill output links
    by_dst = {}
    for l in links:
        by_dst.setdefault((l[1], l[2]), []).append(l[0])
    for n in nodes:
        for si, o in enumerate(n["outputs"]):
            o["links"] = by_dst.get((n["id"], si), [])
    return {
        "id": "robot-removal-qwen-sam3",
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": lid,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def export(out: Path, base: str = "http://127.0.0.1:18188", cfg: dict | None = None) -> Path:
    """Export both phases: <out>.seg.api.json/.seg.json and <out>.edit.api.json/.edit.json."""
    import yaml

    cfg = cfg or yaml.safe_load(Path("robot_removal/config.yaml").read_text())
    out = Path(out)
    base_name = out.with_suffix("")
    seg = build_seg_prompt("INPUT_IMAGE.jpg", "rr_", cfg, (261, 79, 122, 205))
    edit = build_edit_prompt("REFERENCE.png", "rr_", cfg)
    for tag, api in (("seg", seg), ("edit", edit)):
        Path(f"{base_name}.{tag}.api.json").write_text(json.dumps(api, indent=1))
        Path(f"{base_name}.{tag}.json").write_text(json.dumps(to_ui(api, base), indent=1))
    return out


if __name__ == "__main__":
    import sys

    print(export(Path(sys.argv[1] if len(sys.argv) > 1 else "robot_removal/workflow.json")))
