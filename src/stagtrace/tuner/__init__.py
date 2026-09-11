"""Stagtrace tuner client -- everything a calling machine needs, and nothing else.

    from stagtrace import tuner
    study = tuner.connect("<problem-id>", seed=0)

Standard library only: no optimiser, no numpy, no torch, and nothing pinned. The search box, the
budget, the ladder, the evaluation schedule, the acceptance policy and the transfer prior all live
on the server, registered against the problem id before a run starts.

`stagtrace.tuner.wire` is the module; these names are re-exported for convenience so that
`from stagtrace import tuner` is the whole import.
"""
__all__ = ["connect", "RemoteStudy", "Instruction", "TrialPruned", "TrialState"]
__version__ = "0.1.0"


def __getattr__(name):
    """Resolve the public names from `.wire` on first use.

    Importing them eagerly here would put `stagtrace.tuner.wire` in `sys.modules` before
    `python -m stagtrace.tuner.wire` executes it, which makes the interpreter emit a RuntimeWarning
    about unpredictable behaviour on the very first command a new user runs. Lazy resolution keeps
    `from stagtrace import tuner; tuner.connect(...)` working and leaves `-m` clean.
    """
    if name in __all__:
        from . import wire
        return getattr(wire, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
