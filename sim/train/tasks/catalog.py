"""Auto-discovered training task catalog; admission additionally requires qualification evidence.

Every module in ``sim/train/tasks/families/`` (except names starting with ``_``) must export
``TASKS``: a list of ``TaskDefinition``. Family authors never edit a shared registry. A module
that fails to import is skipped and reported in ``DISCOVERY_ERRORS`` (the catalog test fails on
any error), so one family's work in progress cannot break every other family's qualification.
"""

import importlib
import pkgutil
import traceback

from . import families
from .schema import TaskDefinition

DISCOVERY_ERRORS: dict[str, str] = {}
FAMILY_MODULES: dict[str, str] = {}          # task name -> family module name


def discover():
    tasks = {}
    seeds = {}
    DISCOVERY_ERRORS.clear()
    FAMILY_MODULES.clear()
    for info in sorted(pkgutil.iter_modules(families.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module_name = f"{families.__name__}.{info.name}"
        try:
            module = importlib.import_module(module_name)
            exported = list(getattr(module, "TASKS"))
            if not all(isinstance(t, TaskDefinition) for t in exported):
                raise TypeError("TASKS must contain only TaskDefinition instances")
        except Exception:
            DISCOVERY_ERRORS[info.name] = traceback.format_exc(limit=3)
            continue
        for task in exported:
            if task.name in tasks:
                DISCOVERY_ERRORS[info.name] = f"duplicate task name {task.name} (also in {FAMILY_MODULES[task.name]})"
                continue
            clash = set(task.qualification_seeds) & set(seeds)
            if clash:
                other = seeds[min(clash)]
                DISCOVERY_ERRORS[info.name] = (f"qualification seeds of {task.name} collide with {other}; pass "
                                               "qualification_seeds=qualification_seeds_for(name, salt=1)")
                continue
            tasks[task.name] = task
            FAMILY_MODULES[task.name] = info.name
            seeds.update(dict.fromkeys(task.qualification_seeds, task.name))
    return tasks


TRAIN_TASKS = discover()


def load_train_task(name):
    return TRAIN_TASKS[name]


def family_tasks(module):
    """Task names exported by one family module (e.g. ``pilot_blocks``)."""
    return tuple(name for name, owner in FAMILY_MODULES.items() if owner == module)
