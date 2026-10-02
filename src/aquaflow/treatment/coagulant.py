"""Jar-test matching for coagulant dose selection.

A jar test is a credible analog for the target raw water only when BOTH the
raw-water conditions are close AND the settled-water quality it produced was
good.  Candidates are therefore ranked on a blend of raw-water similarity and
settled turbidity, never on raw-water similarity alone.

Weights (documented so ordering is auditable):

* RAW_WEIGHT = 0.6 weight on raw-water similarity, which itself blends
  turbidity 0.70, alkalinity 0.05 and temperature 0.20 (weighted absolute
  deviations from the target).
* SETTLED_WEIGHT = 0.4 weight on settled turbidity (NTU): lower is better.

Deterministic tie-breaking when blended scores compare equal, in order:
lower settled turbidity wins (settled-water quality is the decisive evidence),
then the lower dose wins (cheaper / less chemical loading), then the earlier
row wins (stable input order), so results never depend on set ordering or
floating-point map iteration.
"""
from dataclasses import dataclass

RAW_WEIGHT = 0.6
SETTLED_WEIGHT = 0.4


@dataclass(frozen=True)
class Jar:
    raw_ntu: float
    alkalinity: float
    temp: float
    dose: float
    settled_ntu: float


@dataclass(frozen=True)
class Match:
    jar: Jar
    index: int
    raw_similarity: float
    score: float


def raw_similarity(r: Jar, ntu: float, alk: float, temp: float) -> float:
    """Weighted raw-water distance: turbidity 0.70, alkalinity 0.05, temp 0.20."""
    return abs(r.raw_ntu - ntu) * 0.7 + abs(r.alkalinity - alk) * 0.05 + abs(r.temp - temp) * 0.2


def _score(r: Jar, ntu: float, alk: float, temp: float) -> tuple[float, float]:
    similarity = raw_similarity(r, ntu, alk, temp)
    blended = RAW_WEIGHT * similarity + SETTLED_WEIGHT * r.settled_ntu
    return similarity, blended


def _sort_key(i: int, r: Jar, ntu: float, alk: float, temp: float):
    similarity, blended = _score(r, ntu, alk, temp)
    return (blended, r.settled_ntu, r.dose, i)


def rank_candidates(rows: list[Jar], ntu: float, alk: float, temp: float) -> list[Match]:
    """Return every jar ranked best-first; an empty set is rejected."""
    if not rows:
        raise ValueError("jar tests required")
    ordered = sorted(
        enumerate(rows),
        key=lambda ir: _sort_key(ir[0], ir[1], ntu, alk, temp),
    )
    return [
        Match(jar=r, index=i, raw_similarity=_score(r, ntu, alk, temp)[0], score=_score(r, ntu, alk, temp)[1])
        for i, r in ordered
    ]


def recommend(rows: list[Jar], ntu: float, alk: float, temp: float) -> float:
    """Recommend the dose of the best-ranked jar test."""
    return rank_candidates(rows, ntu, alk, temp)[0].jar.dose
