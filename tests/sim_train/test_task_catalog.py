"""Catalog contracts: a cosmetic rename must not leak held-out semantics."""

from dataclasses import replace

import pytest

from sim.train.tasks import TRAIN_TASKS, load_train_task, validate_catalog
from sim.train.tasks.schema import SemanticSignature, OrderedCompletion


def validation_index():
    return {"tasks": [{"task": "sort_blocks", "instruction":
            "Put the red block on the red plate and the blue block on the blue plate."}],
            "episodes": [{"task": "sort_blocks", "seed": 3}]}


PILOTS = ("block_in_bowl", "block_out_of_bowl", "block_beside_bowl", "blocks_onto_mats_in_order",
          "bar_crosswise_on_mat")


def test_pilots_have_distinct_semantics_and_required_execution_contracts():
    from sim.train.tasks.catalog import DISCOVERY_ERRORS, family_tasks
    assert not DISCOVERY_ERRORS, DISCOVERY_ERRORS
    report = validate_catalog(TRAIN_TASKS.values(), validation_index())
    assert report["accepted"], report
    assert set(family_tasks("pilot_blocks")) == set(PILOTS)
    assert {TRAIN_TASKS[n].family for n in PILOTS} == {
        "container_insertion", "container_removal", "spatial_arrangement",
        "ordered_relocation", "orientation_sensitive_placement"}
    for name, task in TRAIN_TASKS.items():
        assert load_train_task(name) is task
        env_cls, oracle_cls = task.load_classes()
        assert callable(env_cls) and callable(oracle_cls)
        assert task.task_objects and task.semantic_signature.goal
        assert len(task.action_descriptions()) == len(task.action_order)
        assert task.action_descriptions() == task.action_descriptions()
        assert task.qualification_seeds and 3 not in task.qualification_seeds


def test_cosmetic_alias_is_rejected_even_with_new_name_and_instruction():
    task = next(iter(TRAIN_TASKS.values()))
    alias = replace(task, name="renamed_task", instruction="A completely different paraphrase.",
                    semantic_signature=replace(task.semantic_signature,
                        manipulated_objects=tuple("RED " + n.upper() for n in task.semantic_signature.manipulated_objects)))
    report = validate_catalog([task, alias], validation_index())
    assert not report["accepted"]
    assert "duplicate_semantics" in {r["kind"] for r in report["violations"]}


@pytest.mark.parametrize("field,value,kind", [
    ("name", "sort_blocks", "validation_name"),
    ("instruction", " PUT the red block on the red plate and the blue block on the blue plate! ", "validation_instruction"),
    ("qualification_seeds", (3,), "validation_seed"),
])
def test_validation_overlap_rejected(field, value, kind):
    task = replace(next(iter(TRAIN_TASKS.values())), **{field: value})
    report = validate_catalog([task], validation_index())
    assert not report["accepted"]
    assert kind in {r["kind"] for r in report["violations"]}


def test_validation_semantics_are_checked_independently_of_prose():
    task = replace(next(iter(TRAIN_TASKS.values())), semantic_signature=SemanticSignature(
        "spatial_arrangement", ("block", "block"), "on", "matching plates", ("either",)))
    report = validate_catalog([task], validation_index())
    assert not report["accepted"]
    assert "validation_semantics" in {r["kind"] for r in report["violations"]}
    assert report["family_level_isolation"] is False


def test_unknown_validation_semantics_and_malformed_index_fail_closed():
    with pytest.raises(ValueError, match="semantic"):
        validate_catalog(TRAIN_TASKS.values(), {"tasks": [{"task": "unknown"}], "episodes": []})
    with pytest.raises(ValueError):
        validate_catalog(TRAIN_TASKS.values(), {})


@pytest.mark.parametrize("field,value", [("family", ""), ("instruction", " "),
                                        ("action_order", ()), ("task_objects", ())])
def test_required_taxonomy_cannot_be_empty(field, value):
    with pytest.raises(ValueError):
        replace(next(iter(TRAIN_TASKS.values())), **{field: value})


def test_order_requires_real_pick_and_release_and_rejects_reverse_order():
    order = OrderedCompletion(("first", "second"))
    order.observe(held=(), placed=("first", "second"))
    assert not order.complete
    order.observe(held=("second",), placed=())
    order.observe(held=(), placed=("second",))
    assert order.violated and not order.complete
    correct = OrderedCompletion(("first", "second"))
    for name in ("first", "second"):
        correct.observe(held=(name,), placed=())
        correct.observe(held=(), placed=(name,))
    assert correct.complete


def test_unknown_task_lookup_fails():
    with pytest.raises(KeyError):
        load_train_task("sort_blocks")


def test_order_rejects_second_pick_before_first_placement():
    order = OrderedCompletion(("first", "second"))
    order.observe(held=("first",), placed=())
    order.observe(held=("second",), placed=())
    order.observe(held=(), placed=("first", "second"))
    assert order.violated and not order.complete


def test_crosswise_orientation_rejects_upright_but_wrong_yaw():
    import mujoco
    import numpy as np
    from sim.train.tasks.families.pilot_blocks import OrientationEnv, crosswise
    env = OrientationEnv(render_images=False)
    try:
        env.reset(seed=9000)
        xy = env.object_pos("bar")[:2]
        for quat, expected in (((1, 0, 0, 0), False),                                   # lengthwise (x)
                               ((np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)), True),    # crosswise (y)
                               ((np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0), False)):  # on its side
            env.set_object_pose("bar", xy, quat=np.asarray(quat))
            mujoco.mj_forward(env.model, env.data)
            assert crosswise(env, "bar") is expected
    finally:
        env.close()


def test_qualification_requires_fifty_prespecified_seeds_and_95_percent():
    from sim.train.tasks.qualify import qualification_summary
    task = next(iter(TRAIN_TASKS.values()))
    rows = [{"seed": seed, "accepted": i < 48, "reasons": [] if i < 48 else ["goal"]}
            for i, seed in enumerate(task.qualification_seeds)]
    assert qualification_summary(task, rows)["qualified"]
    assert not qualification_summary(task, rows[:-1])["qualified"]
    rows[0]["accepted"] = False
    rows[0]["reasons"] = ["allowed_contacts"]
    summary = qualification_summary(task, rows)
    assert not summary["qualified"] and summary["strict_success_rate"] == 0.94
    assert summary["failure_reasons"] == {"allowed_contacts": 1, "goal": 2}
    rows[0]["seed"] = rows[1]["seed"]
    with pytest.raises(ValueError, match="seed"):
        qualification_summary(task, rows)


def test_missing_stale_or_partial_qualification_cannot_admit_task(tmp_path):
    import json
    from sim.train.tasks.qualify import implementation_hash, require_qualified_task
    task = TRAIN_TASKS[PILOTS[0]]
    with pytest.raises(FileNotFoundError):
        require_qualified_task(task.name, tmp_path)
    rows = [{"seed": seed, "accepted": True, "reasons": []} for seed in task.qualification_seeds]
    report = {"source_hash": "stale", "qualified": True, "overlap_report": {"accepted": True}, "episodes": rows}
    path = tmp_path / f"{task.name}.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification"):
        require_qualified_task(task.name, tmp_path)
    report["source_hash"] = implementation_hash(task)
    report["episodes"] = rows[:49]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification"):
        require_qualified_task(task.name, tmp_path)
    report["episodes"] = rows
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence"):
        require_qualified_task(task.name, tmp_path)


def test_semantic_color_aliases_in_underscored_object_roles_are_equal():
    one = SemanticSignature("ordered_relocation", ("red_block", "blue_block"), "onto", "mats", ("red_block", "blue_block"))
    alias = SemanticSignature("ordered_relocation", ("yellow_block", "green_block"), "onto", "mats", ("yellow_block", "green_block"))
    assert one.normalized() == alias.normalized()


@pytest.mark.parametrize("name", list(TRAIN_TASKS))
def test_real_pilot_reset_policy_and_goals_are_consistent(name):
    import numpy as np
    task = TRAIN_TASKS[name]
    cls, _ = task.load_classes()
    env = cls(render_images=False)
    try:
        env.reset(seed=9000)
        poses = env.object_poses()
        assert env.instruction == task.instruction
        assert not env.success()
        policy = task.physics_policy(env)
        assert policy.settled_frames >= 30
        assert sorted(g.obj for g in env.goals) == sorted(task.task_objects)
        assert policy.grasp_contact is not None
        assert env.goal_regions()
        owners = {env.model.body(int(b)).name for b in env.model.geom_bodyid}
        assert all(s in owners or s in env._body for _, supports in policy.support_bodies for s in supports)
        env.reset(seed=9000)
        assert env.object_poses() == poses
        env.reset(seed=9001)
        assert any(not np.array_equal(env.object_pose(n), poses[n]) for n in task.task_objects)
    finally:
        env.close()
