"""Scripted doubles for the cleaner golden (``tests/test_golden_cleaner.py``).

Everything here is deterministic and offline: a judge keyed by the record markers inside the prompt
it receives, an Assessor keyed the same way, an embedder whose vectors are exact fractions, a clock
that advances one tick per read, and a rubric whose prose and version do not depend on the packaged
rubric artifact.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tabularium.engine.assessor import ContradictionVerdict as ContradictionVerdict
from tabularium.engine.assessor import (
    PanelDisagreement,
    Trace,
    Verdict,
    compute_confidence,
)
from tabularium.engine.judge import JudgeUnavailable
from tabularium.engine.rubric import (
    CliSeat,
    FalsityRubric,
    JudgesConfig,
    RelevanceRubric,
    RevisitRubric,
    Rubric,
    RubricParameters,
)
from tests.support.doubles import TestClock
from tests.support.fake_assessor import EchoAssessor

Fault = Literal["unavailable", "unexpected", "invalid"]
VerdictSpec = Mapping[str, object] | Literal["invalid"]


@dataclass(frozen=True)
class ByAxis:
    """One marker answered differently by the axis whose prompt carries it.

    ``clean()`` shows the same newcomer to the relevance and the falsity axes and an old record to
    the revisit axis, and each schema takes different fields.
    """

    relevance: VerdictSpec | None = None
    falsity: VerdictSpec | None = None
    revisit: VerdictSpec | None = None


ItemSpec = VerdictSpec | ByAxis

_SCOPE_RELEVANCE_GRADES = ("shelve-clear", "shelve-lean", "keep-lean", "keep-clear")
_ITEM_HEADING = re.compile(r"^## (\d+)\.$", re.MULTILINE)
_CLUSTER_TAG = re.compile(r"<c(\d+)(?:\.(\d))?>")
_VARIANTS = ((0.6, 0.8), (0.8, 0.6), (0.28, 0.96), (0.96, 0.28))
_UNTAGGED_BASE = 64
_UNTAGGED_WIDTH = 32


class ScriptError(BaseException):
    """A scripted double was asked something its scenario did not script.

    Derives from ``BaseException`` so ``judge_batch`` and ``ask`` cannot swallow it as a status.
    """


class TickingClock(TestClock):
    """A ``TestClock`` whose every ``now()`` read advances it by ``tick``.

    A read whose value lands in a stored timestamp shows where in the run it happened, so the golden
    pins which code reads the clock and in what order. ``peek`` reads without advancing.
    """

    def __init__(
        self,
        start: datetime.datetime,
        *,
        tick: datetime.timedelta = datetime.timedelta(seconds=1),
    ) -> None:
        super().__init__(start)
        self._tick = tick

    def now(self) -> datetime.datetime:
        current = super().now()
        self.advance(self._tick)
        return current

    def peek(self) -> datetime.datetime:
        return super().now()


class ClusterEmbedder:
    """Exact-arithmetic vectors: texts whose last tag is ``<cN>`` or ``<cN.V>`` share cluster N.

    ``V`` picks one of four unit vectors in the cluster's plane (default 0). Two texts of one
    cluster have cosine between 0.5376 and 1.0 depending on their variants (variants 0 and 1 give
    0.96), and two texts in different clusters have cosine exactly 0. An untagged text is a
    hash-derived vector on the dimensions no cluster uses, so it is orthogonal to every cluster and
    unrelated to every other untagged text. Only addition, multiplication, division and ``sqrt``
    touch the values, so no platform ``libm`` call can move a stored similarity.
    """

    dim = 96
    model = "golden/cluster"

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tags = _CLUSTER_TAG.findall(text)
        if not tags:
            digest = hashlib.shake_256(text.encode()).digest(_UNTAGGED_WIDTH)
            raw = [byte - 127.5 for byte in digest]
            norm = math.sqrt(sum(value * value for value in raw))
            vector[_UNTAGGED_BASE : _UNTAGGED_BASE + _UNTAGGED_WIDTH] = [
                value / norm for value in raw
            ]
            return vector

        cluster, variant = tags[-1]
        first, second = _VARIANTS[int(variant or 0)]
        vector[2 * int(cluster)] = first
        vector[2 * int(cluster) + 1] = second
        return vector

    async def embed(self, text: str) -> list[float]:
        return self.vector(text)


def _split_items(prompt: str) -> list[str]:
    body = prompt.split("# Your batch", 1)[1].split("# Reply format", 1)[0]
    parts = _ITEM_HEADING.split(body)
    return [parts[position].strip() for position in range(2, len(parts), 2)]


def _axis_of(segment: str) -> Literal["relevance", "falsity", "revisit"]:
    if "N (newer, created" in segment:
        return "falsity"
    if "### History (input to weigh" in segment:
        return "revisit"
    return "relevance"


class ScriptedJudge:
    """A judge that answers each batch item by the marker its content carries.

    ``items`` maps a marker substring to the verdict fields for that item (``index`` is added). A
    spec of ``"invalid"`` makes any prompt containing the item unparseable, which is how a record
    the judge cannot answer is modelled. ``faults[n]`` overrides the whole answer to call number
    ``n`` (0-based, per judge), and ``fault_padding`` lengthens the text of a raised fault.
    ``on_call`` runs before each answer, so a scenario can change the store while the judges are
    busy.
    """

    def __init__(
        self,
        identity: str,
        items: Mapping[str, ItemSpec],
        *,
        faults: Sequence[Fault | None] = (),
        fault_padding: str = "",
        on_call: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._identity = identity
        self._items = dict(items)
        self._faults = tuple(faults)
        self._fault_padding = fault_padding
        self._on_call = on_call
        self.prompts: list[str] = []
        self.call_faults: list[Fault | None] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        number = len(self.prompts)
        self.prompts.append(prompt)
        fault = self._faults[number] if number < len(self._faults) else None
        self.call_faults.append(fault)
        if self._on_call is not None:
            await self._on_call(number)

        if fault == "unavailable":
            raise JudgeUnavailable(
                f"{self._identity} scripted outage on call {number}{self._fault_padding}"
            )
        if fault == "unexpected":
            raise RuntimeError(
                f"{self._identity} scripted crash on call {number}{self._fault_padding}"
            )
        if fault == "invalid":
            return "this is not json"

        verdicts: list[dict[str, object]] = []
        for index, segment in enumerate(_split_items(prompt), start=1):
            spec = self._spec_for(segment)
            if spec == "invalid":
                return "this is not json"
            verdicts.append({"index": index, **spec})
        return json.dumps({"verdicts": verdicts})

    def _spec_for(self, segment: str) -> VerdictSpec:
        for marker, spec in self._items.items():
            if marker not in segment:
                continue
            if not isinstance(spec, ByAxis):
                return spec
            axis_spec = getattr(spec, _axis_of(segment))
            if axis_spec is None:
                raise ScriptError(f"{self._identity}: {marker} has no spec for {_axis_of(segment)}")
            result: VerdictSpec = axis_spec
            return result
        raise ScriptError(f"{self._identity}: no scripted marker in item {segment[:90]!r}")


def shelf(
    grade: str,
    why: str | None = None,
    *,
    keep: str | None = None,
    unclear: bool = False,
) -> dict[str, object]:
    """One relevance or revisit item: ``unclear`` is always sent because the schema is strict."""
    item: dict[str, object] = {
        "grade": grade,
        "why": why if why is not None else f"{grade} because the scripted gate says so",
        "unclear": unclear,
    }
    if keep is not None:
        item["keep"] = keep
    return item


def replace(
    reason: str = "changed",
    why: str | None = None,
    *,
    carry: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "verdict": "replace",
        "reason": reason,
        "why": why if why is not None else f"replace ({reason}): the newer record overtook it",
    }
    if carry is not None:
        item["carry"] = carry
    return item


def coexist(why: str | None = None) -> dict[str, object]:
    return {"verdict": "coexist", "why": why if why is not None else "both still stand"}


def _trace() -> Trace:
    return Trace(
        model="golden-assessor",
        wording_hash="golden-hash",
        latency_ms=0,
        input_tokens=None,
        output_tokens=None,
        request_id=None,
    )


def relevance_pick(pick: str, confidence: float = 0.95) -> Verdict:
    other = (1.0 - confidence) / 3
    odds = {grade: (confidence if grade == pick else other) for grade in _SCOPE_RELEVANCE_GRADES}
    return Verdict(
        method="relevance",
        pick=pick,
        odds=odds,
        confidence=compute_confidence(odds),
        trace=_trace(),
    )


def contradiction_pick(
    pick: Literal["coexist", "correction", "changed"], confidence: float = 1.0
) -> ContradictionVerdict:
    reason_class: Literal["correction", "changed"] | None = (
        pick if pick in ("correction", "changed") else None
    )
    remainder = (1 - confidence) / 2
    odds = dict.fromkeys(("coexist", "correction", "changed"), remainder)
    odds[pick] = confidence
    return ContradictionVerdict(
        pick="coexist" if pick == "coexist" else "replace",
        reason_class=reason_class,
        odds=odds,
        confidence=compute_confidence(odds),
        trace=_trace(),
    )


def breadth_pick(pick: Literal["owner_wide", "project_specific"]) -> Verdict:
    return Verdict(
        method="breadth",
        pick=pick,
        odds={pick: 1.0},
        confidence=1.0,
        trace=_trace(),
    )


class ScriptedAssessor(EchoAssessor):
    """An Assessor answering by marker substring.

    ``relevance`` is keyed by the record text, ``contradiction`` by the older record's text and
    ``breadth`` by the newcomer's text. A scripted exception instance is raised, which
    ``admission.ask`` turns into ``Unavailable``. ``log`` records every call in order.
    """

    def __init__(
        self,
        *,
        relevance: Mapping[str, Verdict | BaseException] | None = None,
        contradiction: Mapping[str, ContradictionVerdict | BaseException] | None = None,
        breadth: Mapping[str, Verdict | BaseException] | None = None,
        default_relevance: Verdict | None = None,
    ) -> None:
        super().__init__("unrelated")
        self._relevance_by_marker = dict(relevance or {})
        self._contradiction_by_marker = dict(contradiction or {})
        self._breadth_by_marker = dict(breadth or {})
        self._default_relevance = default_relevance
        self.log: list[dict[str, object]] = []

    async def relevance(self, text: str) -> Verdict:
        self.log.append({"method": "relevance", "text": text})
        scripted = _lookup(self._relevance_by_marker, text)
        if isinstance(scripted, BaseException):
            raise scripted
        if scripted is not None:
            return scripted
        if self._default_relevance is not None:
            return self._default_relevance
        raise ScriptError(f"no scripted relevance answer for {text[:90]!r}")

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        self.log.append(
            {
                "method": "contradiction",
                "existing": existing,
                "incoming": incoming,
                "panel": panel.model_dump(mode="json") if panel is not None else None,
            }
        )
        scripted = _lookup(self._contradiction_by_marker, existing)
        if isinstance(scripted, BaseException):
            raise scripted
        if scripted is not None:
            return scripted
        raise ScriptError(f"no scripted contradiction answer for {existing[:90]!r}")

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        self.log.append({"method": "breadth", "text": text, "source_scope": source_scope})
        scripted = _lookup(self._breadth_by_marker, text)
        if isinstance(scripted, BaseException):
            raise scripted
        if scripted is not None:
            return scripted
        raise ScriptError(f"no scripted breadth answer for {text[:90]!r}")


def _lookup[T](table: Mapping[str, T], text: str) -> T | None:
    for marker, answer in table.items():
        if marker in text:
            return answer
    return None


def golden_rubric(
    *,
    batch_cap: int = 56,
    revisit_batch_cap: int = 12,
    revisit_pass_cap: int = 48,
    report_render_cap: int = 5,
) -> Rubric:
    """A rubric whose version, prose and judge seats are set here, not read from the package."""
    seat = CliSeat(
        model="golden-model",
        cli="claude",
        binary="/usr/bin/golden-claude",
        timeout=30.0,
    )
    return Rubric(
        version="golden-rubric-v1",
        parameters=RubricParameters(
            k=5,
            similarity_floor=0.80,
            batch_cap=batch_cap,
            unclear_ceiling=0.10,
            pool_limit=1000,
            report_render_cap=report_render_cap,
            revisit_batch_cap=revisit_batch_cap,
            revisit_pass_cap=revisit_pass_cap,
            revisit_min_age_days=30,
            revisit_interval_days=90,
            revisit_stable_multiplier=2.0,
            revisit_contested_multiplier=0.5,
            revisit_max_interval_days=360,
        ),
        relevance=RelevanceRubric(
            scale=("shelve-clear", "shelve-lean", "keep-lean", "keep-clear"),
            anchor="golden relevance anchor",
            prose="Golden relevance standard: grade each record against the admission gate.",
        ),
        revisit=RevisitRubric(prose="Golden revisit rules: weigh the history, do not grade it."),
        falsity=FalsityRubric(
            verdicts=("replace", "coexist"),
            anchor="golden falsity anchor",
            prose="Golden falsity standard: does the newer record replace the older one.",
        ),
        judges=JudgesConfig(seat_a=seat, seat_b=seat, backstop=seat),
    )
