"""Five candidate pilots; admission additionally requires qualification evidence."""

from .schema import SemanticSignature, TaskDefinition


def _task(name, instruction, family, objects, relation, goal, order, descriptions, env):
    return TaskDefinition(name, instruction, family,
        SemanticSignature(family, tuple("rectangular block" if n == "bar" else "block" for n in objects),
                          relation, goal, order), objects, order, descriptions,
        f"sim.train.tasks.pilots:{env}", "sim.train.tasks.pilots:PilotOracle")


TRAIN_TASKS = {task.name: task for task in (
    _task("block_in_basket", "Put the block inside the basket.", "container_insertion", ("block",),
          "inside", "basket", ("block",), ("Pick up the block and lower it into the basket.",), "InsertEnv"),
    _task("block_out_of_basket", "Take the block out of the basket and put it on the mat.", "container_removal", ("block",),
          "out of basket onto", "mat", ("block",), ("Lift the block out of the basket and place it on the mat.",), "RemoveEnv"),
    _task("block_beside_bowl", "Place the block to the right of the bowl, leaving a gap.", "spatial_arrangement", ("block",),
          "right of separated", "bowl", ("block",), ("Pick up the block and place it to the right of the bowl.",), "BesideEnv"),
    _task("blocks_row_in_order", "Move the red block to the near mat, then the blue block to the far mat.", "ordered_relocation", ("red_block", "blue_block"),
          "on in temporal order", "near mat then far mat", ("red_block", "blue_block"),
          ("Move the red block onto the near mat.", "Then move the blue block onto the far mat."), "OrderedEnv"),
    _task("bar_crosswise_on_mat", "Place the long block on the mat with its long side running left to right.", "orientation_sensitive_placement", ("bar",),
          "on crosswise", "mat with long axis along table y", ("bar",),
          ("Pick up the long block, turn it crosswise, and place it on the mat.",), "OrientationEnv"),
)}


def load_train_task(name):
    return TRAIN_TASKS[name]
