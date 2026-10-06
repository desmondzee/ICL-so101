"""Spending ledger: caps hold under concurrency, unresolved requests keep their reservation."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json

import pytest

from sim.train import budget
from sim.train.budget import BudgetError, Ledger, reservation_usd


def _reserve(ledger, key, amount=0.09, **kw):
    kw.setdefault("phase", "bulk")
    kw.setdefault("first_attempt", False)
    kw.setdefault("run_id", "r")
    return ledger.reserve(key, amount, **kw)


def test_concurrent_thread_reservations_never_exceed_the_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda i: _reserve(ledger, f"k{i}", cap_usd=1.0)[0], range(60)))
    summary = ledger.summary(1.0)
    assert sum(results) == 11  # floor(1.00 / 0.09)
    assert summary["committed_usd"] <= 1.0
    assert summary["requests"] == 11


def _process_worker(args):
    path, start = args
    ledger = Ledger(path)
    return sum(_reserve(ledger, f"p{start + i}", cap_usd=0.5)[0] for i in range(10))


def test_concurrent_process_reservations_never_exceed_the_cap(tmp_path):
    path = str(tmp_path / "ledger.json")
    with ProcessPoolExecutor(4) as pool:
        granted = sum(pool.map(_process_worker, [(path, i * 100) for i in range(4)]))
    document = json.loads((tmp_path / "ledger.json").read_text())
    assert granted == len(document["requests"]) == 5  # floor(0.50 / 0.09)
    assert budget.totals(document)["committed_usd"] <= 0.5


def test_absolute_cap_cannot_be_raised(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.json", hard_cap_usd=121)
    ledger = Ledger(tmp_path / "l.json")
    ok, reason = _reserve(ledger, "big", amount=119.0, cap_usd=500)
    assert ok
    assert _reserve(ledger, "more", amount=1.5, cap_usd=500) == (False, "hard_cap")


def test_unresolved_requests_keep_their_reservation(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    for key in ("reserved", "submitting", "submitted", "uncertain", "failed"):
        assert _reserve(ledger, key)[0]
    ledger.mark_submitting("submitting")
    ledger.mark_submitting("submitted")
    ledger.mark_submitted("submitted", {"request_id": "r1"})
    ledger.mark_submitting("uncertain")
    ledger.mark_uncertain("uncertain", "TimeoutError during POST")
    ledger.mark_submitting("failed")
    ledger.mark_submitted("failed", {"request_id": "r2"})
    ledger.mark_failed("failed", "model error")
    assert ledger.summary()["committed_usd"] == pytest.approx(5 * 0.09)
    # An uncertain request cannot be resubmitted or released by the code paths.
    with pytest.raises(BudgetError):
        ledger.mark_submitting("uncertain")
    with pytest.raises(BudgetError):
        ledger.release_unsent("uncertain", "x")
    with pytest.raises(BudgetError):
        _reserve(ledger, "uncertain")
    # Crash recovery: 'reserved' never POSTed (released), 'submitting' becomes uncertain.
    assert _reserve(ledger, "crash_reserved")[0]
    assert _reserve(ledger, "crash_submitting")[0]
    ledger.mark_submitting("crash_submitting")
    changed = ledger.recover_interrupted()
    assert "crash_submitting" in changed["uncertain"] and "submitting" in changed["uncertain"]
    assert "crash_reserved" in changed["released"] and "reserved" in changed["released"]
    statuses = {k: e["status"] for k, e in ledger.snapshot()["requests"].items()}
    assert statuses["crash_submitting"] == statuses["uncertain"] == "uncertain"
    assert ledger.summary()["committed_usd"] == pytest.approx(5 * 0.09)  # 3 uncertain + submitted + failed


def test_operator_resolution_of_an_uncertain_post(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    for key in ("a", "b"):
        _reserve(ledger, key)
        ledger.mark_submitting(key)
        ledger.mark_uncertain(key, "timeout")
    ledger.resolve_uncertain("a", handle={"request_id": "found", "status_url": "s", "response_url": "r"})
    ledger.resolve_uncertain("b", not_submitted_evidence="absent from fal request history")
    assert ledger.get("a")["status"] == "submitted"
    assert ledger.get("b")["status"] == "released"
    assert ledger.summary()["committed_usd"] == pytest.approx(0.09)


def test_completion_reconciles_and_releases_surplus(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    _reserve(ledger, "k", amount=0.09)
    ledger.mark_submitting("k")
    ledger.mark_submitted("k", {"request_id": "r"})
    ledger.mark_completed("k", 0.075, "estimate:output_duration")
    assert ledger.summary()["committed_usd"] == pytest.approx(0.075)
    ledger.reconcile("k", 0.08, "fal usage export")
    assert ledger.summary()["committed_usd"] == pytest.approx(0.08)


def test_pilot_cap_retry_reserve_and_run_budget(tmp_path):
    ledger = Ledger(tmp_path / "l.json", pilot_cap_usd=0.2)
    assert _reserve(ledger, "p1", phase="pilot")[0]
    assert _reserve(ledger, "p2", phase="pilot")[0]
    assert _reserve(ledger, "p3", phase="pilot") == (False, "pilot_cap")
    assert _reserve(ledger, "b1", phase="bulk")[0]  # bulk is not limited by the pilot cap
    # First attempts may use only 75% of the cap: 1.0 cap -> 0.75.
    ledger2 = Ledger(tmp_path / "l2.json")
    granted = [_reserve(ledger2, f"f{i}", cap_usd=1.0, first_attempt=True)[1] for i in range(10)]
    assert granted.count("ok") == 8 and granted[8] == "retry_reserve"
    assert _reserve(ledger2, "retry", cap_usd=1.0)[0]  # retries can use the reserve
    ledger3 = Ledger(tmp_path / "l3.json")
    assert [_reserve(ledger3, f"x{i}", run_budget_usd=0.2)[1] for i in range(3)] == ["ok", "ok", "run_budget"]
    assert _reserve(ledger3, "other_run", run_id="r2", run_budget_usd=0.2)[0]


def test_reservation_is_conservative_around_the_price_change():
    early = datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert reservation_usd(5.0, at=early) == 0.09  # (5 + 0.7) s * $0.015 = 0.0855 -> 0.09
    assert reservation_usd(5.0, at=budget.PROMO_END) == 0.15  # 5.7 * 0.025 = 0.1425
    assert reservation_usd(5.0, at=budget.PROMO_END - timedelta(hours=1)) == 0.15  # queue slack
    assert reservation_usd(5.0, at=early, override=0.02) == 0.12
    assert budget.price_per_second(early) == 0.015
    assert budget.price_per_second(datetime(2026, 10, 16, tzinfo=timezone.utc)) == 0.025
