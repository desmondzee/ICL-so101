"""Task registry: name -> "module:EnvClass:OracleClass". Add one line per task."""

import importlib

TASKS = {
    "sort_blocks": "sim.val.tasks.sort_blocks:SortBlocksEnv:SortBlocksOracle",
    "stack_bowls": "sim.val.tasks.stack_bowls:StackBowlsEnv:StackBowlsOracle",
    "mug_on_plate": "sim.val.tasks.mug_on_plate:MugOnPlateEnv:MugOnPlateOracle",
    "mugs_in_microwave": "sim.val.tasks.mugs_in_microwave:MugsInMicrowaveEnv:MugsInMicrowaveOracle",
    "pan_on_stove": "sim.val.tasks.pan_on_stove:PanOnStoveEnv:PanOnStoveOracle",
}


def load(name):
    module, env_cls, oracle_cls = TASKS[name].split(":")
    m = importlib.import_module(module)
    return getattr(m, env_cls), getattr(m, oracle_cls)
