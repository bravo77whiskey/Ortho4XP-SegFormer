"""Run-scoped diagnostic output shared by overlay stages and their loaders."""

from contextvars import ContextVar
from functools import wraps
from inspect import signature


# Standalone loaders retain their usual output. Overlay entry points establish
# their own setting, including in inference threads and texture subprocesses.
_verbose = ContextVar("overlay_verbose_log", default=True)


def verbose_print(*args, **kwargs):
    """Print diagnostics when verbose; return whether anything was printed."""
    if _verbose.get():
        print(*args, **kwargs)
        return True
    return False


def with_verbose_logging(function):
    """Scope diagnostics to a run's verbose_log argument or stage's run object."""
    call_signature = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if "verbose_log" in bound.arguments:
            enabled = bound.arguments["verbose_log"]
        else:
            stage = bound.arguments["self"]
            enabled = getattr(stage, "run", stage).verbose_log
        token = _verbose.set(bool(enabled))
        try:
            return function(*args, **kwargs)
        finally:
            _verbose.reset(token)

    return wrapped
