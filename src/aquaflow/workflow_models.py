"""Input contracts for cross-module operational workflows.

The workflow layer deliberately keeps transport concerns separate from domain
rules.  Each request is normalized once, so downstream modules can reason
about typed records and preserve an audit-friendly input snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkflowEnvelope:
    name: str
    payload: dict[str, Any]
    received_at: datetime


_REQUIRED: dict[str, tuple[str, ...]] = {
    "ct_assurance": ("segments", "dose"),
    "coagulant_match": ("jars", "target"),
    "filter_backwash": ("filters", "now", "offline", "max_offline"),
    "reservoir_age": ("state", "hours"),
    "network_isolation": ("nodes", "pipes", "closed", "sources"),
    "zone_pressure": ("nodes", "low", "high"),
    "pump_dispatch": ("pumps", "history", "required", "max_kw"),
    "booster_failover": ("units", "required", "failed"),
    "custody_chain": ("origin", "events"),
    "collector_schedule": ("requests", "collectors"),
    "lab_revision": ("results", "sample", "analyte"),
    "meter_balance": ("inputs", "outputs", "known_loss"),
    "leak_localization": ("signals",),
    "flush_waves": ("hydrants", "required", "max_parallel"),
    "chemical_active_mass": ("lots",),
    "confined_permit": ("permit", "at", "max_age"),
    "outage_impact": ("customers", "nodes"),
    "incident_escalation": ("incident", "rules", "now"),
    "topology_revision": ("sequence", "revisions"),
    "regulatory_window": ("readings", "window_hours", "at"),
}


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy(v) for v in value)
    return value


def _validate(name: str, normalized: dict[str, Any]) -> None:
    """Workflow-specific structural validation beyond field presence."""
    if name == "coagulant_match":
        jars = normalized["jars"]
        if not isinstance(jars, list) or not jars:
            # An empty jar-test set cannot support a dose recommendation.
            raise ValueError("jar tests required")
        target = normalized["target"]
        jar_fields = ("raw_ntu", "alkalinity", "temp", "dose", "settled_ntu")
        for i, jar in enumerate(jars):
            if not isinstance(jar, dict):
                raise TypeError(f"jar {i} must be a mapping")
            missing = [key for key in jar_fields if key not in jar]
            if missing:
                raise ValueError(f"jar {i} missing fields: " + ",".join(missing))
        target_fields = ("ntu", "alkalinity", "temp")
        missing = [key for key in target_fields if key not in target]
        if missing:
            raise ValueError("target missing fields: " + ",".join(missing))


def normalize(name: str, payload: dict[str, Any], received_at: datetime | None = None) -> WorkflowEnvelope:
    if name not in _REQUIRED:
        raise ValueError("unknown workflow")
    if not isinstance(payload, dict):
        raise TypeError("workflow payload must be a mapping")
    missing = [key for key in _REQUIRED[name] if key not in payload]
    if missing:
        raise ValueError("missing workflow fields: " + ",".join(missing))
    normalized = _copy(payload)
    normalized["_contract"] = "v1"
    _validate(name, normalized)
    normalized["_required"] = _REQUIRED[name]
    return WorkflowEnvelope(name, normalized, received_at or datetime.now().astimezone())
