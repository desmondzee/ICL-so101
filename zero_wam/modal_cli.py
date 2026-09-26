"""Run the Modal CLI with credentials from the local, ignored .env file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values


def main() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    values = dotenv_values(env_file) if env_file.is_file() else {}
    env = os.environ.copy()
    token_id = env.get("MODAL_TOKEN_ID") or values.get("MODAL_TOKEN_ID") or values.get(
        "MODAL_API_KEY"
    )
    token_secret = (
        env.get("MODAL_TOKEN_SECRET")
        or env.get("MODAL_SECRET_TOKEN")
        or values.get("MODAL_TOKEN_SECRET")
        or values.get("MODAL_SECRET_TOKEN")
    )
    if not token_id or not token_secret:
        raise SystemExit(
            "Modal requires a token ID and secret. MODAL_API_KEY may supply "
            "the ID; MODAL_SECRET_TOKEN may supply the secret."
        )
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret
    if "HF_TOKEN" not in env and values.get("HF_TOKEN"):
        env["HF_TOKEN"] = values["HF_TOKEN"]
    modal_executable = Path(sys.executable).with_name("modal")
    if not modal_executable.is_file():
        raise SystemExit(f"Modal CLI not installed next to {sys.executable}")
    os.execve(str(modal_executable), [str(modal_executable), *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
