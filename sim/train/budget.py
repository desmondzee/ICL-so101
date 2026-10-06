"""Durable fal spending ledger with conservative, atomic per-request reservations.

The ledger is one JSON file (default ``<root>/generation/ledger.json``) updated only
inside an exclusive ``fcntl`` lock on a sibling ``.ledger.lock`` file and published with
fsync + atomic rename, so concurrent threads and processes see a serial history.

Accounting rules (all amounts in USD):

* A request is *committed* (counts against every cap) unless it is ``released``
  (its POST provably never left this machine, or the server refused it with a 4xx
  before queueing) or it has been reconciled to an explicit billed amount.
* ``reserved``, ``submitting``, ``submitted``, ``uncertain`` and ``failed`` requests
  keep their full reservation. An uncertain POST is never resubmitted; only an
  operator ``resolve`` (attach the found request id, or confirm it was not sent)
  can change it.
* ``completed`` requests are charged their reconciled cost (estimated from the
  output duration, or a billed amount imported later), which may release surplus.

Caps: an absolute ceiling of $120 (``ABSOLUTE_CAP_USD``), an optional lower
per-call cap, a pilot cap (default $5) over ``phase == "pilot"`` requests, a
per-run budget over requests of one ``run_id`` and an initial retry reserve: first
attempts may only use ``(1 - retry_reserve_fraction) * cap``.

Price: ``price_per_second`` follows the H3 Max Turbo 480P schedule ($0.015/s
promotional, $0.025/s afterwards). The promotion "ends October 15"; the switch is
taken conservatively at 2026-10-14T10:00Z (midnight Oct 15 in UTC+14), and a
reservation uses the highest price within ``QUEUE_SLACK`` of submission. An
explicit override replaces the schedule.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterator

from .store import atomic_write_json

ABSOLUTE_CAP_USD = 120.0
DEFAULT_PILOT_CAP_USD = 5.0
DEFAULT_RETRY_RESERVE = 0.25
PROMO_PRICE, REGULAR_PRICE = 0.015, 0.025
PROMO_END = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
QUEUE_SLACK = timedelta(hours=6)
OUTPUT_OVERRUN_S = 0.7  # fal: "the output can run up to about 0.7 s longer than requested"
LEDGER_SCHEMA_VERSION = 1

# Statuses whose reservation still counts. "released" frees it; "completed" and any
# reconciled request count their reconciled amount instead.
HOLDING = ("reserved", "submitting", "submitted", "uncertain", "failed")
STATUSES = HOLDING + ("completed", "released")
_THREAD_LOCK = threading.Lock()


class BudgetError(RuntimeError):
    pass


class LedgerCorruption(BudgetError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def price_per_second(at: datetime | None = None, override: float | None = None) -> float:
    if override is not None:
        if not (isinstance(override, (int, float)) and math.isfinite(override) and override > 0):
            raise ValueError("price override must be a positive number")
        return float(override)
    at = at or _now()
    return PROMO_PRICE if at < PROMO_END else REGULAR_PRICE


def reservation_usd(duration_s: float, *, at: datetime | None = None, override: float | None = None,
                    margin: float = 1.0) -> float:
    """Conservative charge for one request: highest price in the queue-slack window,
    the documented output overrun, a margin, rounded up to whole cents."""
    if not duration_s > 0 or margin < 1:
        raise ValueError("duration must be positive and margin >= 1")
    at = at or _now()
    price = max(price_per_second(at, override), price_per_second(at + QUEUE_SLACK, override))
    return math.ceil((duration_s + OUTPUT_OVERRUN_S) * price * margin * 100 - 1e-9) / 100


def _charge(entry: dict) -> float:
    if entry.get("reconciled_usd") is not None:
        return float(entry["reconciled_usd"])
    if entry["status"] == "released":
        return 0.0
    return float(entry["reserved_usd"])


def totals(document: dict) -> dict:
    """Committed spend by status / phase / run, recomputed from the requests."""
    out = {"committed_usd": 0.0, "pilot_committed_usd": 0.0, "first_attempt_committed_usd": 0.0,
           "unresolved_usd": 0.0, "reconciled_usd": 0.0, "by_status": {}, "by_run": {}}
    for entry in document["requests"].values():
        charge = _charge(entry)
        out["committed_usd"] += charge
        if entry.get("phase") == "pilot":
            out["pilot_committed_usd"] += charge
        if entry.get("first_attempt"):
            out["first_attempt_committed_usd"] += charge
        if entry["status"] in ("reserved", "submitting", "submitted", "uncertain"):
            out["unresolved_usd"] += charge
        if entry.get("reconciled_usd") is not None:
            out["reconciled_usd"] += charge
        out["by_status"][entry["status"]] = out["by_status"].get(entry["status"], 0) + 1
        run = entry.get("run_id")
        out["by_run"][run] = round(out["by_run"].get(run, 0.0) + charge, 6)
    for name in ("committed_usd", "pilot_committed_usd", "first_attempt_committed_usd", "unresolved_usd", "reconciled_usd"):
        out[name] = round(out[name], 6)
    return out


class Ledger:
    def __init__(self, path: str | Path, *, hard_cap_usd: float = ABSOLUTE_CAP_USD,
                 pilot_cap_usd: float = DEFAULT_PILOT_CAP_USD):
        if not 0 < hard_cap_usd <= ABSOLUTE_CAP_USD:
            raise ValueError(f"hard cap must be in (0, {ABSOLUTE_CAP_USD}]")
        if not 0 <= pilot_cap_usd <= hard_cap_usd:
            raise ValueError("pilot cap must be within the hard cap")
        self.path = Path(path)
        self.hard_cap_usd = float(hard_cap_usd)
        self.pilot_cap_usd = float(pilot_cap_usd)

    # ----- storage --------------------------------------------------------------------------------------
    def _fresh(self) -> dict:
        return {"schema_version": LEDGER_SCHEMA_VERSION, "created_at": _now().isoformat(),
                "absolute_cap_usd": ABSOLUTE_CAP_USD, "requests": {}}

    def _read(self) -> dict:
        if not self.path.exists():
            return self._fresh()
        try:
            document = json.loads(self.path.read_text())
            if document.get("schema_version") != LEDGER_SCHEMA_VERSION or not isinstance(document.get("requests"), dict):
                raise ValueError("unknown ledger schema")
            for key, entry in document["requests"].items():
                if entry.get("status") not in STATUSES or not isinstance(entry.get("reserved_usd"), (int, float)):
                    raise ValueError(f"invalid ledger entry {key}")
            return document
        except (OSError, ValueError) as exc:
            raise LedgerCorruption(f"cannot read ledger {self.path}: {exc}") from exc

    @contextmanager
    def _transaction(self, write: bool = True) -> Iterator[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCK, (self.path.parent / f".{self.path.name}.txn.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                document = self._read()
                yield document
                if write:
                    document["updated_at"] = _now().isoformat()
                    atomic_write_json(self.path, document)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def snapshot(self) -> dict:
        if not self.path.exists():  # read-only callers (dry runs) never create files
            return self._fresh()
        with self._transaction(write=False) as document:
            return json.loads(json.dumps(document))

    def get(self, key: str) -> dict | None:
        return self.snapshot()["requests"].get(key)

    def summary(self, cap_usd: float | None = None) -> dict:
        document = self.snapshot()
        cap = self._cap(cap_usd)
        t = totals(document)
        return {**t, "cap_usd": cap, "remaining_usd": round(cap - t["committed_usd"], 6),
                "pilot_cap_usd": self.pilot_cap_usd, "requests": len(document["requests"])}

    def _cap(self, cap_usd: float | None) -> float:
        cap = self.hard_cap_usd if cap_usd is None else min(float(cap_usd), self.hard_cap_usd)
        if not cap > 0:
            raise ValueError("cap must be positive")
        return cap

    @staticmethod
    def _event(entry: dict, status: str, **extra) -> None:
        entry["history"].append({"at": _now().isoformat(), "status": status, **extra})
        entry["status"] = status

    def _entry(self, document: dict, key: str, allowed: tuple[str, ...]) -> dict:
        entry = document["requests"].get(key)
        if entry is None:
            raise BudgetError(f"unknown request {key}")
        if entry["status"] not in allowed:
            raise BudgetError(f"request {key} is {entry['status']}, expected one of {allowed}")
        return entry

    # ----- reservations ---------------------------------------------------------------------------------
    def reserve(self, key: str, amount_usd: float, *, phase: str, first_attempt: bool, run_id: str,
                cap_usd: float | None = None, run_budget_usd: float | None = None,
                retry_reserve_fraction: float = DEFAULT_RETRY_RESERVE, meta: dict | None = None) -> tuple[bool, str]:
        """Atomically reserve ``amount_usd`` if every cap still holds. Returns (ok, reason)."""
        if phase not in ("pilot", "bulk"):
            raise ValueError("phase must be pilot or bulk")
        if not (amount_usd > 0 and math.isfinite(amount_usd)):
            raise ValueError("reservation must be positive")
        if not 0 <= retry_reserve_fraction < 1:
            raise ValueError("retry reserve fraction must be in [0, 1)")
        cap = self._cap(cap_usd)
        with self._transaction() as document:
            if key in document["requests"]:
                raise BudgetError(f"request {key} already exists in the ledger ({document['requests'][key]['status']})")
            t = totals(document)
            eps = 1e-9
            if t["committed_usd"] + amount_usd > cap + eps:
                return False, "hard_cap"
            if phase == "pilot" and t["pilot_committed_usd"] + amount_usd > self.pilot_cap_usd + eps:
                return False, "pilot_cap"
            if first_attempt and t["first_attempt_committed_usd"] + amount_usd > cap * (1 - retry_reserve_fraction) + eps:
                return False, "retry_reserve"
            if run_budget_usd is not None and t["by_run"].get(run_id, 0.0) + amount_usd > run_budget_usd + eps:
                return False, "run_budget"
            entry = {"key": key, "status": "reserved", "reserved_usd": round(amount_usd, 6), "reconciled_usd": None,
                     "phase": phase, "first_attempt": bool(first_attempt), "run_id": run_id,
                     "request_id": None, "handle": None, "history": [], **(meta or {})}
            self._event(entry, "reserved", cap_usd=cap)
            document["requests"][key] = entry
            return True, "ok"

    def mark_submitting(self, key: str) -> None:
        """Durably record that a POST is about to leave; a crash after this is uncertain."""
        with self._transaction() as document:
            self._event(self._entry(document, key, ("reserved",)), "submitting")

    def mark_submitted(self, key: str, handle: dict) -> None:
        if not handle.get("request_id"):
            raise ValueError("submission handle needs a request_id")
        with self._transaction() as document:
            entry = self._entry(document, key, ("submitting",))
            entry["request_id"], entry["handle"] = handle["request_id"], handle
            self._event(entry, "submitted", request_id=handle["request_id"])

    def mark_uncertain(self, key: str, reason: str) -> None:
        with self._transaction() as document:
            self._event(self._entry(document, key, ("submitting",)), "uncertain", reason=reason)

    def release_unsent(self, key: str, reason: str) -> None:
        """Only for requests that provably never reached fal's queue."""
        with self._transaction() as document:
            self._event(self._entry(document, key, ("reserved", "submitting")), "released", reason=reason)

    def mark_failed(self, key: str, reason: str) -> None:
        """fal reported failure; the reservation is kept until billed usage is reconciled."""
        with self._transaction() as document:
            self._event(self._entry(document, key, ("submitted",)), "failed", reason=reason)

    def mark_completed(self, key: str, cost_usd: float, source: str) -> None:
        with self._transaction() as document:
            entry = self._entry(document, key, ("submitted", "completed"))
            if entry["status"] == "completed":
                return
            entry["reconciled_usd"] = round(float(cost_usd), 6)
            entry["cost_source"] = source
            self._event(entry, "completed", cost_usd=entry["reconciled_usd"], source=source)

    def reconcile(self, key: str, billed_usd: float, source: str) -> None:
        """Replace a settled request's charge with billed usage (e.g. from the fal dashboard)."""
        if not (billed_usd >= 0 and math.isfinite(billed_usd)):
            raise ValueError("billed amount must be >= 0")
        with self._transaction() as document:
            entry = self._entry(document, key, ("completed", "failed"))
            entry["reconciled_usd"] = round(float(billed_usd), 6)
            entry["cost_source"] = source
            entry["history"].append({"at": _now().isoformat(), "status": entry["status"],
                                     "reconciled_usd": entry["reconciled_usd"], "source": source})

    def resolve_uncertain(self, key: str, *, handle: dict | None = None, not_submitted_evidence: str | None = None) -> None:
        """Operator resolution of an uncertain POST after checking fal's request history."""
        if (handle is None) == (not_submitted_evidence is None):
            raise ValueError("give exactly one of handle or not_submitted_evidence")
        with self._transaction() as document:
            entry = self._entry(document, key, ("uncertain",))
            if handle is not None:
                if not handle.get("request_id"):
                    raise ValueError("handle needs a request_id")
                entry["request_id"], entry["handle"] = handle["request_id"], handle
                self._event(entry, "submitted", request_id=handle["request_id"], resolved="operator")
            else:
                self._event(entry, "released", reason=f"operator: {not_submitted_evidence}")

    def recover_interrupted(self) -> dict:
        """After a crash: 'reserved' never started its POST (released); 'submitting' is uncertain."""
        changed = {"released": [], "uncertain": []}
        with self._transaction() as document:
            for key, entry in document["requests"].items():
                if entry["status"] == "reserved":
                    self._event(entry, "released", reason="interrupted before POST")
                    changed["released"].append(key)
                elif entry["status"] == "submitting":
                    self._event(entry, "uncertain", reason="interrupted during POST; delivery unknown")
                    changed["uncertain"].append(key)
        return changed
