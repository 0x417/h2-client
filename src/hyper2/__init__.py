"""hyper2 client — everything a customer's machine needs, and nothing else.

    import hyper2
    study = hyper2.connect("<problem-id>", seed=0)

Standard library only: no optimiser, no numpy, no torch, and nothing pinned. The search box, the
budget, the ladder, the evaluation schedule, the acceptance policy and the transfer prior all live
on the server, registered against the problem id before a run starts.

`hyper2.wire` is the module; these names are re-exported for convenience so that `import hyper2`
is the whole import.
"""
__all__ = ["connect", "RemoteStudy", "Instruction", "TrialPruned", "TrialState"]
__version__ = "0.1.0"


def __getattr__(name):
    """Resolve the public names from `.wire` on first use.

    Importing them eagerly here would put `hyper2.wire` in `sys.modules` before
    `python -m hyper2.wire` executes it, which makes the interpreter emit a RuntimeWarning about
    unpredictable behaviour on the very first command a new user runs. Lazy resolution keeps
    `import hyper2; hyper2.connect(...)` working and leaves `-m` clean.
    """
    if name in __all__:
        from . import wire
        return getattr(wire, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
