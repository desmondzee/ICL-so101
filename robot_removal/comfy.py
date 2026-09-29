"""Minimal ComfyUI API client: upload, submit, poll, download outputs."""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18188", timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def system_stats(self) -> dict:
        return requests.get(f"{self.base}/system_stats", timeout=self.timeout).json()

    def upload_image(self, path: Path, name: str | None = None, overwrite: bool = False) -> str:
        """Upload into ComfyUI's input dir; returns the name to pass to LoadImage."""
        name = name or f"{uuid.uuid4().hex}_{path.name}"
        with open(path, "rb") as f:
            r = requests.post(
                f"{self.base}/upload/image",
                files={"image": (name, f)},
                data={"overwrite": "true" if overwrite else "false", "type": "input"},
                timeout=120,
            )
        r.raise_for_status()
        return r.json()["name"]

    def submit(self, prompt: dict) -> str:
        r = requests.post(f"{self.base}/prompt", json={"prompt": prompt, "client_id": "robot_removal"}, timeout=self.timeout)
        if r.status_code != 200:
            raise ComfyError(f"submit failed {r.status_code}: {r.text[:2000]}")
        body = r.json()
        if body.get("node_errors"):
            raise ComfyError(f"node errors: {json.dumps(body['node_errors'])[:2000]}")
        return body["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        r = requests.get(f"{self.base}/history/{prompt_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get(prompt_id)

    def wait(self, prompt_id: str, poll: float = 2.0, timeout: float = 3600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.history(prompt_id)
            if h is not None:
                status = h.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    return h
                msgs = status.get("messages", [])
                if status.get("status_str") == "error" or any(m[0] == "execution_error" for m in msgs):
                    raise ComfyError(f"execution error: {json.dumps(msgs)[-3000:]}")
            time.sleep(poll)
        try:
            requests.post(f"{self.base}/interrupt", timeout=10)
        except requests.RequestException:
            pass
        raise ComfyError(f"timed out waiting for {prompt_id}")

    def view(self, filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
        r = requests.get(f"{self.base}/view?{q}", timeout=120)
        r.raise_for_status()
        return r.content

    def output_files(self, history: dict) -> list[dict]:
        """Flat list of {node_id, filename, subfolder, type} for every saved image."""
        files = []
        for node_id, out in history.get("outputs", {}).items():
            for im in out.get("images", []):
                files.append({"node_id": node_id, **im})
        return files
