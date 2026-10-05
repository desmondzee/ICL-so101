"""Durable manifests and checked workflow transitions.

Only state.json is replaced. Manifests and uniquely named human-attempt records
are immutable. A store-wide advisory lock serializes writers and readers;
atomic JSON publication also uses a per-file lock for standalone callers.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .model import EpisodeKey, EpisodeManifest, EpisodeRecord, EpisodeState


class StoreError(ValueError):
    """A persisted episode cannot satisfy the requested operation."""


class InvalidTransition(StoreError):
    pass


class EvidenceMismatch(StoreError):
    pass


class ManifestConflict(StoreError):
    pass


class StoreCorruption(StoreError):
    pass


_PATH = tuple(state for state in EpisodeState if state is not EpisodeState.REJECTED)
_NEXT = dict(zip(_PATH, _PATH[1:]))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _evidence_hashes(evidence: dict[str, Any]) -> dict[str, str]:
    if not isinstance(evidence, dict) or any(not isinstance(key, str) for key in evidence):
        raise ValueError("evidence must be a JSON object with string keys")
    return {key: _digest(value) for key, value in evidence.items()}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256_file(path: str | Path) -> str:
    """Hash bytes in bounded chunks rather than loading videos into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    """Flush and fsync a sibling .tmp before publishing with os.replace.

    Set overwrite=False for final manifests/artifacts. Cooperating writers use
    the same per-file lock, so two immutable writes cannot replace one another.
    """
    payload = _json_bytes(value) + b"\n"
    path = Path(path)
    with _lock(path.with_name(f".{path.name}.lock")):
        if not overwrite and (path.exists() or path.is_symlink()):
            raise FileExistsError(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=unique_keys,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise StoreCorruption(f"cannot read {path}: {exc}") from exc


def _legal(old: EpisodeState, new: EpisodeState) -> bool:
    if old in (EpisodeState.ACCEPTED, EpisodeState.REJECTED):
        return False
    return new is EpisodeState.REJECTED or _NEXT[old] is new


class EpisodeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def episode_dir(self, key: EpisodeKey) -> Path:
        return self.root / "candidates" / key.task / f"episode_{key.seed}"

    def manifest_path(self, key: EpisodeKey) -> Path:
        return self.episode_dir(key) / "manifest.json"

    def state_path(self, key: EpisodeKey) -> Path:
        return self.episode_dir(key) / "state.json"

    def _locked(self):
        return _lock(self.root / ".store.lock")

    def _check_artifacts(self, key: EpisodeKey, hashes: dict[str, str]) -> None:
        if not isinstance(hashes, dict):
            raise EvidenceMismatch("artifact_hashes must map relative paths to SHA-256 hashes")
        directory = self.episode_dir(key).resolve()
        for name, expected in hashes.items():
            if not isinstance(name, str) or not name or not _valid_hash(expected):
                raise EvidenceMismatch("invalid artifact path/hash")
            relative = Path(name)
            artifact = directory / relative
            if relative.is_absolute() or ".." in relative.parts or not artifact.resolve().is_relative_to(directory):
                raise EvidenceMismatch(f"artifact escapes episode directory: {name}")
            try:
                actual = sha256_file(artifact)
            except OSError as exc:
                raise EvidenceMismatch(f"cannot hash artifact {name}: {exc}") from exc
            if actual != expected:
                raise EvidenceMismatch(f"artifact hash mismatch: {name}")

    def _check_evidence(self, key: EpisodeKey, evidence: dict[str, Any]) -> dict[str, str]:
        hashes = _evidence_hashes(evidence)
        self._check_artifacts(key, evidence.get("artifact_hashes", {}))
        return hashes

    def create_candidate(self, manifest: EpisodeManifest) -> EpisodeRecord:
        if not _valid_hash(manifest.config_hash):
            raise ValueError("config_hash must be a lowercase SHA-256 hex digest")
        if not isinstance(manifest.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        document = json.loads(_json_bytes({"schema_version": 1, **asdict(manifest)}))
        with self._locked():
            path = self.manifest_path(manifest.key)
            if path.exists():
                existing = self._load(manifest.key)
                if _json_bytes(_read_json(path)) != _json_bytes(document):
                    raise ManifestConflict(f"episode already exists with different manifest: {manifest.key}")
                return existing
            if self.state_path(manifest.key).exists():
                raise StoreCorruption("state exists without immutable manifest")
            self._check_artifacts(manifest.key, manifest.artifacts)
            atomic_write_json(path, document, overwrite=False)
            atomic_write_json(self.state_path(manifest.key), {
                "schema_version": 1,
                "key": asdict(manifest.key),
                "manifest_hash": sha256_file(path),
                "state": EpisodeState.CANDIDATE.value,
                "history": [],
            }, overwrite=False)
            return self._load(manifest.key)

    def load(self, key: EpisodeKey) -> EpisodeRecord:
        with self._locked():
            return self._load(key)

    def _load(self, key: EpisodeKey) -> EpisodeRecord:
        document = _read_json(self.manifest_path(key))
        state = _read_json(self.state_path(key))
        try:
            if document["schema_version"] != 1 or state["schema_version"] != 1:
                raise ValueError("unknown schema_version")
            if document["key"] != asdict(key) or state["key"] != asdict(key):
                raise ValueError("episode key mismatch")
            if not _valid_hash(document["config_hash"]) or not isinstance(document["metadata"], dict):
                raise ValueError("invalid manifest metadata/config hash")
            if state["manifest_hash"] != sha256_file(self.manifest_path(key)):
                raise ValueError("immutable manifest hash mismatch")
            manifest = EpisodeManifest(key, document["config_hash"], document["metadata"], document["artifacts"])
            self._check_artifacts(key, manifest.artifacts)
            current = EpisodeState.CANDIDATE
            history = state["history"]
            if not isinstance(history, list):
                raise ValueError("history must be a list")
            for entry in history:
                old, new = EpisodeState(entry["old_state"]), EpisodeState(entry["new_state"])
                if old is not current or not _legal(old, new):
                    raise ValueError("invalid state history")
                self._validate_event(key, entry)
                current = new
            if EpisodeState(state["state"]) is not current:
                raise ValueError("state does not match history")
            attempts = []
            for path in sorted((self.episode_dir(key) / "human_attempts").glob("*.json")):
                entry = _read_json(path)
                if entry["attempt_id"] != path.stem:
                    raise ValueError("attempt ID does not match path")
                self._validate_event(key, entry)
                attempts.append(entry)
            return EpisodeRecord(manifest, current, tuple(history), tuple(attempts))
        except EvidenceMismatch:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise StoreCorruption(f"invalid episode {key}: {exc}") from exc

    def _validate_event(self, key: EpisodeKey, entry: dict[str, Any]) -> None:
        stamp = datetime.fromisoformat(entry["timestamp"])
        if stamp.utcoffset() is None:
            raise ValueError("event timestamp must include timezone")
        hashes = self._check_evidence(key, entry["evidence"])
        if hashes != entry["evidence_hashes"]:
            raise ValueError("event evidence hash mismatch")

    def transition(self, key: EpisodeKey, expected: EpisodeState, target: EpisodeState,
                   evidence: dict[str, Any]) -> EpisodeRecord:
        expected, target = EpisodeState(expected), EpisodeState(target)
        with self._locked():
            record = self._load(key)
            hashes = self._check_evidence(key, evidence)
            # A completed edge may be replayed even after later gates advanced.
            for entry in record.history:
                if entry["old_state"] == expected.value and entry["new_state"] == target.value:
                    if entry["evidence"] != evidence or entry["evidence_hashes"] != hashes:
                        raise EvidenceMismatch("completed transition has different evidence")
                    return record
            if record.state is not expected or not _legal(expected, target):
                raise InvalidTransition(f"cannot advance {record.state} via {expected} -> {target}")
            state = _read_json(self.state_path(key))
            state["state"] = target.value
            state["history"].append({
                "timestamp": _timestamp(), "old_state": expected.value, "new_state": target.value,
                "evidence": evidence, "evidence_hashes": hashes,
            })
            atomic_write_json(self.state_path(key), state)
            return self._load(key)

    def record_human_attempt(self, key: EpisodeKey, attempt_id: str,
                             evidence: dict[str, Any]) -> EpisodeRecord:
        """Persist a unique request outcome without rejecting the robot source.

        Human submission/review workflow and budgeting are downstream policies;
        failed outcomes here never advance or regress the episode's quality gates.
        """
        if not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", attempt_id):
            raise ValueError("attempt_id must be a filesystem-safe identifier")
        with self._locked():
            record = self._load(key)
            hashes = self._check_evidence(key, evidence)
            path = self.episode_dir(key) / "human_attempts" / f"{attempt_id}.json"
            if path.exists():
                previous = _read_json(path)
                if previous["evidence"] != evidence or previous["evidence_hashes"] != hashes:
                    raise EvidenceMismatch("human attempt already has different evidence")
                return record
            if record.state not in _PATH[3:8]:
                raise InvalidTransition("human attempts require a nonterminal robot-approved source")
            atomic_write_json(path, {
                "attempt_id": attempt_id, "timestamp": _timestamp(),
                "evidence": evidence, "evidence_hashes": hashes,
            }, overwrite=False)
            return self._load(key)
