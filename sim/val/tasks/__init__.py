"""Task registry: name -> "module:EnvClass:OracleClass". Add one line per task."""

import importlib

TASKS = {
    "sort_blocks": "sim.val.tasks.sort_blocks:SortBlocksEnv:SortBlocksOracle",
}


def load(name):
    module, env_cls, oracle_cls = TASKS[name].split(":")
    m = importlib.import_module(module)
    return getattr(m, env_cls), getattr(m, oracle_cls)
