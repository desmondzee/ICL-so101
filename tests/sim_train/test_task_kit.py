"""Task-authoring kit: discovery, seeds, goal-derived policy, rest pose and motion smoothness."""

import textwrap

import mujoco
import numpy as np
import pytest

from sim.train.tasks import TRAIN_TASKS
from sim.train.tasks import catalog
from sim.train.tasks.base import JAW_BODIES, TRAIN_REST_DEG, TrainOracle, define_task, step_text, then
from sim.train.tasks.families.pilot_blocks import BesideEnv, BesideOracle, OrderedEnv
from sim.train.tasks.schema import qualification_seeds_for


def test_every_family_module_imports_and_seeds_are_disjoint():
    assert not catalog.DISCOVERY_ERRORS, catalog.DISCOVERY_ERRORS
    seen = set()
    for task in TRAIN_TASKS.values():
        seeds = set(task.qualification_seeds)
        assert len(seeds) == 50 and min(seeds) >= 1_000_000, task.name   # never validation/calibration seeds
        assert not seeds & seen, task.name
        seen |= seeds


def test_seed_blocks_are_name_derived_and_salted():
    assert qualification_seeds_for("x") == qualification_seeds_for("x")
    assert qualification_seeds_for("x") != qualification_seeds_for("x", salt=1)
    assert qualification_seeds_for("x") != qualification_seeds_for("y")


def test_discovery_isolates_broken_and_conflicting_family_modules(tmp_path, monkeypatch):
    good = textwrap.dedent('''
        from sim.train.tasks.base import define_task
        from sim.train.tasks.families.pilot_blocks import BesideEnv, BesideOracle
        TASKS = [define_task(name="kit_test_good", instruction=BesideEnv.instruction, family="kit_test",
                             env=BesideEnv, oracle=BesideOracle, objects=("block",), relation="r", goal="g",
                             steps=("Pick up the block.",))]
    ''')
    (tmp_path / "zz_kit_a_good.py").write_text(good)
    (tmp_path / "zz_kit_c_broken.py").write_text("raise RuntimeError('work in progress')\n")
    (tmp_path / "zz_kit_b_dupe.py").write_text(good)          # same task name, same seeds
    (tmp_path / "_zz_kit_private.py").write_text("raise RuntimeError('never imported')\n")
    monkeypatch.setattr(catalog.families, "__path__", [str(tmp_path)])
    try:
        found = catalog.discover()
        assert set(found) == {"kit_test_good"}
        assert set(catalog.DISCOVERY_ERRORS) == {"zz_kit_c_broken", "zz_kit_b_dupe"}
        assert "work in progress" in catalog.DISCOVERY_ERRORS["zz_kit_c_broken"]
        assert catalog.family_tasks("zz_kit_a_good") == ("kit_test_good",)
    finally:
        monkeypatch.undo()
        catalog.discover()
    assert not catalog.DISCOVERY_ERRORS


def test_define_task_rejects_contract_mismatch():
    with pytest.raises(ValueError, match="task_objects"):
        define_task(name="bad", instruction=BesideEnv.instruction, family="f", env=BesideEnv, oracle=BesideOracle,
                    objects=("bar",), relation="r", goal="g", steps=("x",))
    with pytest.raises(ValueError, match="instruction"):
        define_task(name="bad", instruction="Something else.", family="f", env=BesideEnv, oracle=BesideOracle,
                    objects=("block",), relation="r", goal="g", steps=("x",))


def test_action_text_helpers():
    assert step_text("put", "red block", "on", "green mat") == "Pick up the red block and put it on the green mat."
    assert then("Pick up the blue block.") == "Then pick up the blue block."
    ordered = TRAIN_TASKS["blocks_onto_mats_in_order"]
    assert ordered.action_order == ("red_block", "blue_block") and len(ordered.action_text) == 2


def test_policy_is_derived_from_goals_with_bounded_jaw_contact_only():
    env = OrderedEnv(render_images=False)
    try:
        env.reset(seed=9000)
        policy = env.physics_policy()
        allowance = policy.grasp_contact
        assert allowance.robot_bodies == JAW_BODIES
        assert env.table_body in allowance.surfaces and {"green_mat", "yellow_mat"} <= set(allowance.surfaces)
        assert set(policy.distractors) == set(env.free_names) - set(env.task_objects)
        assert dict(policy.support_bodies) == {"red_block": ("green_mat",), "blue_block": ("yellow_mat",)}
        robot = {env.model.body(b).name for b in range(env.model.nbody)
                 if env._in_subtree(b, env.model.body("base").id)}
        for pair in policy.allowed_contacts:      # no robot link is ever whitelisted outright
            assert not set(pair) & (robot - set(JAW_BODIES)), pair
        assert [g.reference for g in policy.goals] == ["green_mat", "yellow_mat"]
    finally:
        env.close()


@pytest.mark.parametrize("seed", [9000, 9001, 9002])
def test_training_rest_pose_is_contact_free(seed):
    env = BesideEnv(render_images=False)
    try:
        env.reset(seed=seed)
        m = env.model
        root = m.body("base").id
        np.testing.assert_allclose(np.degrees(env._get_current_qpos())[:5], TRAIN_REST_DEG[:5], atol=2.5)
        for c in env.data.contact[:env.data.ncon]:
            robot = [env._in_subtree(m.geom_bodyid[g], root) for g in (c.geom1, c.geom2)]
            assert not any(robot), (m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)
    finally:
        env.close()


def _lift_targets(oracle_cls):
    """Commanded joints for a pick-height -> carry-height lift at r = 0.25 m (wrist flex saturates on the way)."""
    from sim.val.oracle import top_down_mat
    env = BesideEnv(render_images=False)
    try:
        env.reset(seed=9000)
        o = oracle_cls(env)
        o.start()
        rot = top_down_mat(0.0)
        start = np.array([0.0388 + 0.25, 0.0, 0.015])
        o.q = o.solve(start, rot)[0]
        o._cmd_pos, o._cmd_rot = start, rot
        return np.array([t[:5] for t in o.move(np.array([start[0], 0.0, 0.10]), rot, label="lift")]), o.dt
    finally:
        env.close()


def test_training_lift_has_no_velocity_step_where_the_validation_oracle_does():
    from sim.val.oracle import Oracle
    for cls, bound in ((TrainOracle, 12.0), (Oracle, None)):
        q, dt = _lift_targets(cls)
        acceleration = np.abs(np.diff(q, 2, axis=0)).max() / dt ** 2
        if bound is None:
            assert acceleration > 15.0     # documents the root cause: a one-frame stop at the wrist-flex limit
        else:
            assert acceleration < bound, acceleration


def test_scanned_object_builds_and_rests_upright_when_cached():
    from so101_nexus import gso_assets
    model_id = "CoQ10"
    if not (gso_assets._CACHE_DIR / model_id / "visual.obj").exists():
        pytest.skip("GSO asset not cached (downloads from the public HF mirror on first use)")
    from sim.train.tasks.assets import Scanned, measure
    result = measure([Scanned("bottle", "gso", model_id, scale=0.5)])["bottle"]
    assert result["upright"] and result["graspable"]
    assert 2.5 < min(result["dims_cm"][:2]) < 5.0
