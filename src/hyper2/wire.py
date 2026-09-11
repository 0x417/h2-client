# ⚠ GENERATED FILE -- DO NOT EDIT HERE.
# Emitted from src/hyper2/wire.py in the hyper2 development repo by
# scripts/build_client_repo.py. Edit the source, re-run the generator, and the
# in-sync test will confirm the two are byte-identical.
#!/usr/bin/env python3
"""The wire client: everything the calling machine needs, and nothing else.

Why it is a top-level module: importing a submodule executes its package's `__init__`, and
`hyper2.client.__init__` is the in-process engine, which pulls in numpy. A client whose only job is
to train and report would then have to install an optimiser it never calls and reconcile that
numpy against whatever the training stack pins. This module imports nothing beyond the standard
library, so no such conflict is possible.

    import hyper2.wire as hyper2        # stdlib only
    study = hyper2.connect("<problem-id>", seed=0)

`hyper2.connect` routes here, and `hyper2.client.remote` re-exports from here. The file is small
enough to vendor: one module and the standard library are the entire client.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from enum import Enum

class TrialPruned(Exception):
    """Raise inside an objective to yield the trial at a pause gate -- Optuna's word, deliberately,
    so existing pruning-aware objectives port by changing one import.

    ⚠ DEFINED HERE AND IMPORTED BY THE ENGINE, not defined in both. Two classes of the same name
    are not the same class: an objective raising the wire module's `TrialPruned` would sail
    straight through the engine's `except TrialPruned`, and a trial that should have paused would
    fail the study instead. `hyper2.client` re-exports this one."""


class TrialState(Enum):
    """Values are the engine's, verbatim -- they are compared and logged as strings."""

    RUNNING = "RUNNING"
    PRUNED = "PRUNED"      # paused-or-culled: a PRUNED trial's readings stay recommendable;
                           # the plan may re-invoke the same CONFIG in a later trial
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"






class _Transient(Exception):
    """A failure that MIGHT go away on its own: an unreachable host, a dropped socket, a 5xx. Not a
    4xx -- a rejected setting or a wrong key is deterministic, and retrying it only delays the same
    error."""


class Instruction:
    """What the scheduler says should happen to this configuration NEXT.

    ⚠ WHY IT EXISTS. `report()` returning `False` conflates two
    different statements -- *this session's work package is done* and *this configuration is
    written off, release its storage* -- and a bool cannot distinguish them. `instruct()` does.

        trial.report(ce, ckpt, tokens=actual)   # pure reporting, no back-channel
        ins = trial.instruct()                  # the explicit instruction
        if not ins:
            if ins.discard:
                shutil.rmtree(out_dir / str(trial.number))     # written off
            break                                              # else: keep it, it resumes

    `__bool__` is "keep training this config now", so the loop structure is unchanged -- which is
    unchanged: only the stop condition moves from `report()` to `instruct()`.
    """

    __slots__ = ("action",)

    def __init__(self, action: str):
        self.action = action                     # "continue" | "pause" | "discard"

    def __bool__(self) -> bool:
        return self.action == "continue"

    @property
    def keep_checkpoint(self) -> bool:
        """Keep this config's directory? True unless it was written off. A paused config is
        normally resumed, and freeing storage on every falsy instruction destroys the checkpoint
        that resume depends on."""
        return self.action != "discard"

    @property
    def discard(self) -> bool:
        return self.action == "discard"

    @property
    def paused(self) -> bool:
        return self.action == "pause"

    def __repr__(self) -> str:
        return f"Instruction({self.action!r})"


class _RemoteTrial:
    """One logical trial, its parameters already drawn by the server."""

    def __init__(self, study, number, params, order, last_step=None):
        self._study = study
        self.number = int(number)
        self.params = dict(params)
        # THE RESUME CONTRACT, AND IT HAS TO COME FROM THE SERVER. `last_step` is Optuna's
        # FrozenTrial meaning -- the last checkpoint a PREVIOUS invocation of this config reached,
        # so a continued session knows where to resume (`start = trial.last_step or 0`). It is not
        # a live counter, and setting it on every `report()` here would have made the two
        # transports mean different things by the same name. The server holds the trial that knows
        # it, so the server sends it.
        self.last_step = None if last_step is None else int(last_step)
        self._asked: set = set()        # knobs the objective actually read; see `optimize`
        # WAS THIS SESSION GRANTED ANY WORK AT ALL? Recorded from the server's own reply, not
        # inferred from how far the driver has iterated: "the plan granted nothing" and "the driver
        # has not reached the loop yet" both show zero orders consumed, and only the first of them
        # is a legitimate `instruct()` before any `report()`.
        self._session_empty = order is None
        self._order = order          # ONE order at a time; the next arrives with report()'s reply
        self._session_over = order is None
        self.state = TrialState.RUNNING
        self.intermediate_values: dict[int, float] = {}
        self.signals: dict[int, dict] = {}
        self.value: float | None = None
        self.tokens_actual = 0.0
        self._continue: bool | None = None
        self._aborted = False

    # -- Optuna-identical suggest signatures ---------------------------------------------------
    @staticmethod
    def _norm(spec):
        """Compare a declaration against a suggest call WITHOUT counting defaults as differences.

        `suggest_float("lr", 3e-5, 4e-4, log=True)` builds `("float", 3e-5, 4e-4, True, None)` --
        five elements, the last being the `step=None` default -- while the natural declaration is
        the four-element `("float", 3e-5, 4e-4, True)`. Comparing the raw tuples rejected a
        declaration that was correct in every element it actually contained, and it did so on the
        FIRST suggest call of the first trial, so the server transport could not run a customer's
        objective at all. Trailing "not set" markers are stripped from both sides, and the numbers
        are compared as floats so `32` and `32.0` are the same bound.
        """
        t = list(spec)
        while t and t[-1] in (None, False, 1):    # step=None, log=False, step=1 -- all "unset"
            t.pop()
        out = []
        for x in t:
            if isinstance(x, (list, tuple)):
                # ⛔ A CHOICE LIST IS A SEQUENCE, AND JSON HAS ONLY ONE OF THEM. The declared space
                # crosses the wire as a LIST and `suggest_categorical` builds a TUPLE, so comparing
                # them raw rejected a declaration correct in every element it contained -- exactly
                # the failure this function already existed to prevent, one level in. Every
                # categorical knob was unusable over the wire: the first suggest call raised.
                out.append(tuple(x))
                continue
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(x)
        return tuple(out)

    def _get(self, name, spec):
        declared = self._study._space.get(name)
        if declared is not None and self._norm(declared) != self._norm(spec):
            raise ValueError(
                f"parameter '{name}' was declared to the server as {tuple(declared)} but suggested "
                f"as {tuple(spec)}. The server drew this configuration against the declared box, "
                f"so this cannot be remapped -- fix the declaration or the suggest call.")
        if name not in self.params:
            raise ValueError(
                f"parameter '{name}' was suggested but never declared in space=. The server draws "
                f"whole configurations, so every knob the objective asks for must be in the box.")
        self._asked.add(name)
        return self.params[name]

    def suggest_float(self, name, low, high, *, step=None, log=False):
        return float(self._get(name, ("float", float(low), float(high), bool(log), step)))

    def suggest_int(self, name, low, high, step=1, log=False):
        return int(self._get(name, ("int", int(low), int(high), bool(log), int(step))))

    def suggest_categorical(self, name, choices):
        return self._get(name, ("cat", tuple(choices)))

    # -- the work order ------------------------------------------------------------------------
    def plan(self):
        """The session's work orders: `(checkpoint, cumulative_target, kind)` triples.

        A GENERATOR, exactly as in-process, and for the same reason: the plan decides the next
        depth from the reading just supplied, so the next order exists only after `report()`.
        Draining this into a list up front truncates the session to its first order -- measured,
        that turned a study that spends 8.000 FE over 21 configurations into one that spent 0.500
        over 4, while still looking like it worked.

        THREE values, not two. Both transports yield the identical shape, so a driver written against
        one runs unchanged against the other.
        """
        while self._order is not None:
            o = self._order
            self._order = None
            yield int(o["ckpt"]), float(o["target"]), str(o["kind"])
        self._session_over = True

    def report(self, value, step, tokens=None, **signals) -> bool:
        """Report one checkpoint. `tokens` is the ACTUAL token count consumed to reach it.

        `tokens=` is named rather than left to `**signals`: a step count times a nominal batch
        size overshoots the real figure, and booking the actual count absorbs that instead of
        leaving it to a correction. It is recorded,
        totalled on the study as `tokens_actual`, and written to the audit log beside the plan's own
        accounting so the two can be compared rather than assumed equal.

        `tokens=` is the CUMULATIVE actual count this configuration has now consumed -- the same
        axis as the `target_tokens` in the work order, not an increment. Cumulative because a
        configuration is PAUSED AND RESUMED: adding a per-segment figure double-counts nothing,
        but adding a cumulative one counts the whole prefix again at every resume, and mixing the
        two silently inflates the ledger. The study keeps the largest value seen per configuration
        and sums those, so a driver that re-reports a checkpoint costs nothing.
        """
        payload = {"trial_number": self.number, "step": int(step), "value": float(value),
                   "signals": signals}
        if tokens is not None:
            payload["tokens"] = float(tokens)
            self.tokens_actual = max(self.tokens_actual, float(tokens))
            self._study._note_tokens(self.number, self.tokens_actual)
        r = self._study._post("report", payload)
        self.intermediate_values[int(step)] = float(value)
        if signals:
            self.signals[int(step)] = dict(signals)
        # THE SERVER'S OWN DIAGNOSIS, RAISED HERE. A reading the plan refuses -- an out-of-range
        # step, say -- kills the study thread, and the honest thing is to hand the caller that
        # traceback immediately rather than let them wait out a timeout and then hear about the
        # network.
        if r.get("error"):
            raise RuntimeError(f"the hyper2 server rejected this reading and the study has "
                               f"stopped:\n{r['error']}")
        self._aborted = bool(r.get("aborted"))
        if "spent" in r:
            self._study._spent_seen = float(r["spent"])    # the budget clock, no extra round trip
        self._order = r.get("order")          # the NEXT order rides back with the reply
        self._continue = bool(r.get("continue"))
        return self._continue

    def instruct(self) -> Instruction:
        """What happens to this configuration next -- the explicit form of `report()`'s bool.

        Call it after `report()`. Falsy means this session is over; `.discard` then says whether
        the configuration was written off (free its storage) or merely paused (keep the directory,
        it will be resumed).
        """
        if self._continue is None:
            if self._session_empty:
                # AN EMPTY SESSION IS LEGITIMATE, not a driver error. The plan sometimes queues a
                # configuration whose entitlement is already satisfied and grants it no work; the
                # server has an explicit branch for it. There is nothing to report, so the honest
                # answer is "paused, keep the checkpoint" -- and raising here made the documented
                # post-loop `if trial.instruct().discard:` crash on exactly that session.
                return Instruction("pause")
            raise RuntimeError("call trial.report(...) before trial.instruct(): the instruction is "
                               "the scheduler's answer to a reading, and no reading has been sent")
        if self._aborted:
            return Instruction("discard")
        if not self._continue or self._order is None:
            # SESSION OVER, CONFIG KEPT. `report()` returning True does not mean there is more work
            # -- the ordinary end of a session is the plan simply not issuing another order, and a
            # driver that reads only the bool never learns the session ended, which is the storage
            # question this method exists to answer.
            return Instruction("pause")
        return Instruction("continue")

    @property
    def aborted(self) -> bool:
        """Was the CONFIG written off, or was the session merely paused? See the in-process
        docstring: freeing storage on every False break destroys every paused config's
        checkpoint and silently disables resume."""
        return self._aborted

    def should_prune(self) -> bool:
        return False


class RemoteStudy:
    """`Study`'s surface over HTTP. `optimize(objective)` -- no n_trials; the plan is the stop
    condition, and the plan lives on the server."""

    @property
    def spent(self) -> float:
        """Budget consumed so far, in the customer's full-run equivalents. Same name, same meaning
        on both transports -- the integration test reached for `study._plan.spent`, which is a
        PRIVATE attribute that simply does not exist on this side, so a driver written against the
        in-process engine crashed the moment it was pointed at a server. Updated from the reply to
        each `report()`, so it is current as of the last reading the driver sent.
        """
        return float(getattr(self, "_spent_seen", 0.0))

    @property
    def space(self) -> dict:
        """The box the SERVER will draw from, `{name: (kind, low, high, log)}`.

        Public because a driver needs it: the objective has to call `suggest_*` once per knob with
        the matching bounds, and without this the only way to know them was to import the engine's
        own box definition -- which is the dependency this module exists to remove. A copy, so a
        caller cannot edit the box the suggest-guard checks against.
        """
        return {k: tuple(v) for k, v in self._space.items()}

    #: The signal name this study judges on. Report it under this name at the checkpoints
    #: `needs_judged_metric` identifies -- NOT at every one:
    #:     if study.needs_judged_metric(ckpt):
    #:         signals[study.target_key] = expensive_eval(model)
    #: (This once said to report it at every one, which was true while `eval_at` was fixed
    #: server-side. Once the caller could choose the schedule that stopped being true, and
    #: following it multiplies the single most expensive thing a driver does.)
    #: Read from the server rather than agreed in prose, because a driver reporting the wrong name
    #: leaves the plan with nothing to measure -- it screens one cohort, stops, and still returns a
    #: recommendation. The client raises on that, but a driver that reads the name never gets there.

    @property
    def run_id(self) -> str:
        """`"<problem>:<seed>"` -- a name for THIS run, derived from the problem and the seed.

        Read-only, and convenient for labelling anything the driver keeps: log lines, and above all
        directories. Two repetitions of one problem differ only in the seed, so a name built from
        the problem alone has them overwrite each other.
        """
        return f"{self.problem}:{self.seed}"

    def needs_judged_metric(self, ckpt) -> bool:
        """Does THIS checkpoint need the expensive, judged metric? Ask; do not translate.

        The loop variable `ckpt` is an opaque 1-based index -- 1, 2, 3, 4 -- while the schedule is
        set in fractions of a run (`eval_at=(0.5, 1.0)`). Making a driver bridge those two
        vocabularies itself is how `if ckpt in study.eval_steps` ends up in an objective: correct,
        exact, and unreadable. This asks the question the driver actually has.

            for ckpt, tokens_target, kind in trial.plan():
                ce = cheap_eval(model)
                signals = {}
                if study.needs_judged_metric(ckpt):
                    signals[study.target_key] = expensive_eval(model)
                trial.report(ce, ckpt, tokens=used, **signals)

        With no schedule registered every checkpoint is an evaluation point, so this returns True
        throughout -- which is the correct default and not a special case.
        """
        steps = self.eval_steps
        return True if not steps else int(ckpt) in steps

    @property
    def ladder(self) -> tuple:
        """Every checkpoint of this study, as fractions of one complete run.

        `plan()` yields the depths in the caller's own unit; this is the same set as fractions, and
        the thing `eval_at` was resolved against. If a requested evaluation depth was not already a
        checkpoint it was inserted, so this may differ from the problem's registered ladder --
        `study.description` says so when it does.
        """
        return tuple(self._ladder) if self._ladder else ()

    @property
    def eval_at(self) -> tuple:
        """The evaluation schedule as FRACTIONS of one complete run -- `(0.5, 1.0)` is halfway and
        at the end. The same thing `eval_steps` gives as checkpoint indices, in the unit a person
        reasons in. Fractions are the only form accepted: see `connect`."""
        return tuple(self._eval_at) if self._eval_at else ()

    @property
    def eval_steps(self) -> tuple:
        """The checkpoint indices at which the judged metric is NEEDED -- `()` means every one.

        The judged metric is usually far more expensive to compute than the cheap signal that
        drives screening, so it is produced only at the checkpoints named here. This is that
        schedule, from the server, so a driver does not have to guess it or pay for it at every
        rung:

            ce = cheap_eval(model)                       # every checkpoint: it drives screening
            score = expensive_eval(model) if ckpt in study.eval_steps else None
            trial.report(ce, ckpt, tokens=used, **({study.target_key: score} if score else {}))

        Reporting the judged metric more often than this is harmless to the search but spends
        compute the ledger does not charge; reporting it less often leaves the plan unable to
        certify a crossing where it intended to.
        """
        return tuple(self._eval_steps) if self._eval_steps else ()

    def _note_tokens(self, number, cumulative) -> None:
        self._tokens_by_trial[int(number)] = float(cumulative)

    @property
    def tokens_actual(self) -> float:
        """Real tokens consumed across the study, from `report(..., tokens=...)`.

        The sum of each configuration's largest reported CUMULATIVE count -- see `report`. Compare
        it with `spent_cost`, which is the plan's own accounting: they answer different questions
        and a gap between them is information, not an error."""
        return float(sum(self._tokens_by_trial.values()))

    @property
    def spent_cost(self) -> float:
        """Budget consumed so far IN THE UNIT THE PROBLEM WAS REGISTERED IN -- tokens, normally.
        `spent` is full-run equivalents; see the in-process docstring for why both are named."""
        return float(self.spent) * float(getattr(self, "_unit", 1.0) or 1.0)

    @property
    def budget_cost(self) -> float:
        """The registered total budget in that same unit. What `spent_cost` counts towards."""
        return float(getattr(self, "_budget_cost", 0.0))

    def __init__(self, base_url, space=None, api_key=None, timeout=900.0, problem=None, **cfg):
        self._spent_seen = 0.0
        self._tokens_by_trial: dict = {}        # config -> its largest reported cumulative count
        self._seq = 0                           # idempotency key for mutating calls; see `_raw`
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._space = {k: tuple(v) for k, v in (space or {}).items()}
        self._timeout = float(timeout)
        self.trials: list[_RemoteTrial] = []
        body = dict(cfg)
        body["api_key"] = api_key
        if problem:
            # THE CUSTOMER FORM: a problem id and nothing else. The box, the budget, the ladder,
            # the schedule, the target, the donor library and every lever are registered on the
            # server. `cfg` is normally empty here; anything in it is checked against that
            # problem's `client_may_set` allowlist SERVER-SIDE and refused by name if it is not on
            # it. Deciding that here would put the policy on the machine that does not own it.
            body["problem"] = str(problem)
        else:
            body["space_spec"] = {k: list(v) for k, v in self._space.items()}
        r = self._raw("POST", "/v1/studies", body)
        self.study_id = r["study_id"]
        self.problem = problem
        self.description = r.get("description")
        # THE SEED IN FORCE, always -- the server fixes one if the caller did not. Use it to namespace
        # anything stored on disk: `trial.number` alone collides across repetitions, so two seeds
        # would share a checkpoint directory and overwrite each other.
        self.seed = r.get("seed")
        self.target_key = str(r.get("target_key") or "structural")
        self._eval_steps = r.get("eval_steps")
        self._eval_at = r.get("eval_at")
        self._ladder = r.get("ladder")
        self._unit = float(r.get("full_run_cost") or 1.0)
        self._budget_cost = float(r.get("budget_cost") or 0.0)
        # THE BOX COMES BACK FROM THE SERVER and becomes what `suggest_*` is checked against. On
        # the problem-id path the client declared nothing, so without this there would be no guard
        # at all between "the objective asked for lr in [1e-5,1e-3]" and "the server drew it from
        # [1e-4,1e-2]" -- plausible numbers, a different search, and no trace in the results. On
        # the harness path the server echoes the same box it was handed, so a disagreement here is
        # a real transport bug and is raised rather than absorbed.
        srv = {k: tuple(v) for k, v in (r.get("space") or {}).items()}
        if srv:
            if self._space:
                for name, spec in self._space.items():
                    got = srv.get(name)
                    if got is None or _RemoteTrial._norm(got) != _RemoteTrial._norm(spec):
                        raise RuntimeError(
                            f"the server's box for '{name}' is {got}, but this client declared "
                            f"{tuple(spec)}. The server draws the configurations, so this would "
                            f"run a different search than the one declared.")
            self._space = srv

    # -- transport ------------------------------------------------------------------------------
    #: How many times a request is retried, and the base of the backoff in seconds. A study runs
    #: for hours over a network neither end controls; one dropped packet must not cost the compute spent
    #: so far. Bounded, because an endpoint that is genuinely down should be reported, not waited
    #: for indefinitely: 5 attempts at 1s, 2s, 4s, 8s is ~15s of patience.
    RETRIES = 5
    BACKOFF = 1.0

    def _raw(self, method, path, body=None):
        """One request, retried on TRANSIENT failure only.

        ⚠ RETRYING A MUTATING CALL IS ONLY SAFE BECAUSE THE SERVER IS IDEMPOTENT. `report` and
        `session_end` advance the plan; a retry of a request whose REPLY was lost would otherwise
        report the same reading twice and move the schedule twice -- silently, and only under the
        network conditions nobody tests. Each carries a monotonic `seq`, and a repeat of the last
        one returns the stored reply instead of being applied again.

        4xx is NOT retried: a rejected setting or a wrong key is deterministic, and retrying it
        turns an instant, legible error into fifteen seconds of silence followed by the same error.
        """
        import time as _time
        if body is not None and path.rsplit("/", 1)[-1] in ("report", "session_end"):
            self._seq += 1
            body = dict(body, seq=self._seq)
        last = None
        for attempt in range(self.RETRIES):
            try:
                return self._raw_once(method, path, body)
            except _Transient as e:
                last = e.__cause__ or e
                if attempt == self.RETRIES - 1:
                    break
                _time.sleep(self.BACKOFF * (2 ** attempt))
        raise RuntimeError(
            f"cannot reach the hyper2 server at {self._base} after {self.RETRIES} attempts "
            f"({last}). Start it with `python -m hyper2.server`, and "
            f"check the tailnet and the API key. If the server is still up, the study is alive on "
            f"it and these calls are idempotent, so retrying the same step resumes rather than "
            f"double-reporting.") from None

    def _raw_once(self, method, path, body=None):
        data = json.dumps(body if body is not None else {}).encode()
        headers = {"Content-Type": "application/json"}
        if getattr(self, "_api_key", None):
            # ON EVERY REQUEST, not only the one that carries it in its body. The study-creation
            # call happened to include the key as a field, which made an authenticated endpoint
            # look reachable right up until the first `next` came back 401 -- after the customer
            # had already started training.
            headers["X-API-Key"] = str(self._api_key)
        req = urllib.request.Request(self._base + path, data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if 500 <= e.code < 600:
                raise _Transient(f"{e.code}: {detail}") from e     # the server may recover
            raise RuntimeError(f"hyper2 server {e.code} on {method} {path}: {detail}") from None
        except urllib.error.URLError as e:
            raise _Transient(f"unreachable: {e.reason}") from e
        except (TimeoutError, OSError) as e:                       # a dropped socket mid-study
            raise _Transient(f"{type(e).__name__}: {e}") from e

    def _post(self, leaf, body=None):
        return self._raw("POST", f"/v1/studies/{self.study_id}/{leaf}", body)

    # -- the loop --------------------------------------------------------------------------------
    def optimize(self, objective) -> None:
        """Serve the server's work orders until the plan is done.

        The LOOP THAT DECIDES ANYTHING is the server's -- it runs the real
        `hyper2.client.Study.optimize`, with all of its backfill and non-termination guards, and
        hands each work order here. This side only executes them, which is exactly the division a
        customer's trainer has.
        """
        while True:
            wo = self._post("next")
            if wo.get("done"):
                if wo.get("error"):
                    raise RuntimeError(f"hyper2 server study failed:\n{wo['error']}")
                # ⛔ A STUDY THAT NEVER SAW ITS TARGET METRIC DID NOT RUN, IT ONLY RETURNED.
                # Reporting the wrong signal name -- `report(ce, step, ce=ce)` against a problem
                # judged on `structural` -- leaves the plan with nothing to measure. It screens one
                # cohort, stops, and hands back a recommendation: 4 configurations and 0.75 of a
                # budget of 8, with no exception anywhere. That is a driver bug the customer must
                # find during their integration test, so it is raised rather than returned.
                why = str(wo.get("reason") or "")
                if "no target readings received" in why:
                    raise RuntimeError(
                        f"this study never received its target metric, so nothing could be "
                        f"measured and the search stopped after one cohort.\n  {why}\n"
                        f"  Report it as a SIGNAL on every checkpoint, e.g. "
                        f"trial.report(ce_loss, step=ckpt, structural=my_structural_score).")
                return
            t = _RemoteTrial(self, wo["trial_number"], wo["params"], wo.get("order"),
                             last_step=wo.get("last_step"))
            self.trials.append(t)
            try:
                v = objective(t)
            except TrialPruned:
                t.state = TrialState.PRUNED
                self._post("session_end", {"pruned": True})
                continue
            except Exception:
                t.state = TrialState.FAILED
                self._post("session_end", {"pruned": True})
                raise
            # ⛔ EVERY DECLARED KNOB MUST HAVE BEEN READ. The server draws a WHOLE configuration
            # and names all of it in the recommendation; an objective that reads three of thirteen
            # trains a model that responds to three while the search ranges over thirteen, and the
            # recommended configuration then names ten values the customer never applied. Nothing
            # else catches it: the suggest guard only checks the knobs that ARE asked for, so the
            # run completes and looks correct. Checked on the first session, so it is found before
            # any GPU time.
            missing = sorted(set(self._space) - t._asked)
            if missing and t._asked:
                raise ValueError(
                    f"the objective read {len(t._asked)} of {len(self._space)} declared knobs and "
                    f"never asked for {missing}. The server drew values for those and will name "
                    f"them in the recommendation, so a model trained without them is not the "
                    f"configuration being scored. Add a suggest_* call for each, or register a "
                    f"problem whose box matches the one being trained.")
            t.value = None if v is None else float(v)
            t.state = TrialState.COMPLETE
            self._post("session_end", {"value": t.value})

    # -- results -----------------------------------------------------------------------------------
    @property
    def recommendation(self) -> dict:
        """The config the ENGINE recommends, plus its readout. The product's answer."""
        return self._raw("GET", f"/v1/studies/{self.study_id}/recommendation")

    @property
    def best_trial(self) -> _RemoteTrial:
        done = [t for t in self.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if not done:
            raise ValueError("no trial completed")
        return min(done, key=lambda t: t.value)


def connect(problem: str, *, seed: int | None = None, target: float | None = None,
            eval_at=None, eval_steps=None, budget: float | None = None,
            tokens_per_run: float | None = None,
            server: str | None = None, api_key: str | None = None, timeout: float = 900.0):
    """Open a study against a hyper2 endpoint. See `hyper2.client.connect` for the full contract.

    The caller's entire configuration surface. `seed` and `target` are accepted because neither
    changes what is searched; anything else is registered on the server and refused BY NAME here.
    `server`/`api_key` fall back to `$HYPER2_SERVER` / `$HYPER2_API_KEY`.
    """
    url = server or os.environ.get("HYPER2_SERVER")
    if not url:
        raise ValueError("connect() needs the endpoint: pass server='https://...' or set "
                         "$HYPER2_SERVER. This is a connection detail, not a search setting.")
    extra = {k: v for k, v in (("seed", seed), ("target", target)) if v is not None}
    if tokens_per_run is not None:
        # COST OF ONE COMPLETE TRAINING RUN, IN THE CALLER'S UNIT. Stated by the caller, because only the calling stack can measure it. It sets the unit for everything: the budget, the checkpoint targets
        # `plan()` yields, and the train-out the plan reserves for the winner.
        extra["full_run_cost"] = float(tokens_per_run)
        extra["full_run"] = float(tokens_per_run)
    if budget is not None:
        # TOTAL TOKENS FOR THE WHOLE SEARCH, screening AND training the recommendation out to full
        # depth. Total-inclusive on purpose: it is the number the caller actually pays.
        extra["total_budget"] = float(budget)
    if eval_at is not None:
        # WHERE THE JUDGED METRIC IS PRODUCED, as FRACTIONS of one complete run: `(0.5, 1.0)` is
        # halfway and at the end. Fractions only -- a token count is refused by name, because the
        # same count is a different depth for every caller and would move silently whenever
        # tokens_per_run was re-measured. A depth that is not already a checkpoint is ADDED as one
        # (a judged reading can only be taken where training pauses) and `study.description` says
        # so; nothing is ever moved to a nearby depth quietly.
        extra["eval_at"] = ([float(eval_at)] if isinstance(eval_at, (int, float))
                            else [float(x) for x in eval_at])
    if eval_steps is not None:
        # THE EVALUATION SCHEDULE, SET BY THE CALLER. The judged metric is typically expensive, so which
        # checkpoints it is paid at are the caller's decision -- the engine adapts to whatever the caller
        # can afford. Sent as the engine's `target_rungs`: 1-based checkpoint indices into this
        # problem's ladder.
        extra["target_rungs"] = [int(x) for x in eval_steps]
    return RemoteStudy(url, problem=problem, api_key=api_key or os.environ.get("HYPER2_API_KEY"),
                       timeout=float(timeout), **extra)


# ------------------------------------------------------------------------------------------------
# `python -m hyper2.wire --check` -- the first thing a customer should run, before any training.
#
# It answers the three questions that otherwise get answered by a failed GPU run: can I reach the
# endpoint, is my key accepted, and which problems will it run. Kept in this module so the
# stdlib-only client is self-sufficient: no extra script to ship, nothing to install.

def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m hyper2.wire",
                                 description="check a hyper2 endpoint and list what it will run")
    ap.add_argument("--check", action="store_true", help="reach the endpoint and list its problems")
    ap.add_argument("--server", default=os.environ.get("HYPER2_SERVER"))
    ap.add_argument("--api-key", default=os.environ.get("HYPER2_API_KEY"))
    a = ap.parse_args(argv)
    if not a.server:
        print("set HYPER2_SERVER (and HYPER2_API_KEY), or pass --server", file=sys.stderr)
        return 2
    base = a.server.rstrip("/")
    for path, need_key in (("/v1/health", False), ("/v1/problems", True)):
        req = urllib.request.Request(base + path)
        if a.api_key and need_key:
            req.add_header("X-API-Key", a.api_key)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            print(f"⛔ {path}: HTTP {e.code}"
                  + ("  -- the API key was rejected or missing" if e.code == 401 else ""),
                  file=sys.stderr)
            return 1
        except Exception as e:                                   # noqa: BLE001
            print(f"⛔ cannot reach {base}{path}: {e}", file=sys.stderr)
            return 1
        if path == "/v1/health":
            print(f"endpoint {base}  reachable, auth={'on' if body.get('auth') else 'off'}")
        else:
            ps = body.get("problems") or []
            print(f"{len(ps)} problem(s) this endpoint will run:")
            for p in ps:
                print(f"  {p['id']:24s} judged on {p.get('target_key','?')!r}"
                      f"   the caller may set {sorted(set(p.get('client_may_set') or []) | {'seed'})}")
    print("\n✅ transport and key are good. Next: python examples/integration_test.py")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main())
