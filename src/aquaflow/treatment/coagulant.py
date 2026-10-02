"""Jar-test matching for coagulant dose recommendation.

Candidates are ranked by a composite score that combines raw-water
similarity with settled-water quality, so a jar that merely resembles the
raw water cannot outrank one with clearly better settled turbidity.

Score weights (per unit of each measurement):
  * raw turbidity gap   0.7 / NTU
  * alkalinity gap      0.05 / (mg/L)
  * temperature gap     0.2 / degC
  * settled turbidity   1.0 / NTU  (direct outcome evidence, so each NTU
    of settled turbidity counts at least as much as raw-water mismatch)

Ties on the composite score are broken deterministically: lower settled
turbidity first (stronger outcome evidence), then lower dose (less
chemical, less sludge).  Any remaining tie is resolved by input order,
so repeated runs over the same request always publish the same dose.
"""
from dataclasses import dataclass

RAW_NTU_WEIGHT = 0.7
ALKALINITY_WEIGHT = 0.05
TEMP_WEIGHT = 0.2
SETTLED_WEIGHT = 1.0


@dataclass(frozen=True)
class Jar:
    raw_ntu: float
    alkalinity: float
    temp: float
    dose: float
    settled_ntu: float


def _score(jar: Jar, ntu: float, alk: float, temp: float) -> float:
    similarity = (
        RAW_NTU_WEIGHT * abs(jar.raw_ntu - ntu)
        + ALKALINITY_WEIGHT * abs(jar.alkalinity - alk)
        + TEMP_WEIGHT * abs(jar.temp - temp)
    )
    return similarity + SETTLED_WEIGHT * jar.settled_ntu


def choose(rows: list[Jar], ntu: float, alk: float, temp: float) -> Jar:
    """Return the winning jar; deterministic on ties."""
    if not rows:
        raise ValueError("jar tests required")
    return min(rows, key=lambda r: (_score(r, ntu, alk, temp), r.settled_ntu, r.dose))


def recommend(rows: list[Jar], ntu: float, alk: float, temp: float) -> float:
    return choose(rows, ntu, alk, temp).dose
