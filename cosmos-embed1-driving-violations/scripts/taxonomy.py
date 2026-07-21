"""Single source of truth for violation classes, captions, and glue defaults.

Every script (metadata, inference, glue, eval, demo) imports this module or
loads an override JSON of the same shape via --taxonomy (the self-test uses an
override with HMDB pseudo-violations). INVARIANT: caption strings here are the
ONLY caption strings anywhere in the pipeline.

kind:   "action" (brief, intermittent)  -> short smoothing window
        "state"  (persistent condition) -> long smoothing window
glue defaults per class: median filter window (odd, in stride steps),
threshold (pre-tuning placeholder; 14_tune_thresholds.py overwrites via
thresholds.json), min_consec (hysteresis: event needs >= k consecutive
above-threshold windows).
"""

import json
from pathlib import Path

NEGATIVE_CLASS = "no_violation"

CLASSES = [
    {"id": "trainer_phone_use",
     "caption": "the instructor in the passenger seat is using a mobile phone",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 2},
    {"id": "trainer_no_seatbelt",
     "caption": "the instructor in the passenger seat is not wearing a seatbelt",
     "kind": "state", "window": 9, "threshold": 0.25, "min_consec": 3},
    {"id": "driver_hands_away",
     "caption": "the driver has taken both hands off the steering wheel",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 2},
    {"id": "driver_sleeping",
     "caption": "the driver is sleeping or dozing off at the wheel",
     "kind": "state", "window": 5, "threshold": 0.25, "min_consec": 3},
    {"id": "loose_items",
     "caption": "loose items are lying on the dashboard or center console",
     "kind": "state", "window": 9, "threshold": 0.25, "min_consec": 3},
    {"id": "trainer_intervention",
     "caption": "the instructor reaches over and grabs the steering wheel, hand brake or gear lever",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 1},
    {"id": NEGATIVE_CLASS,
     "caption": "a normal driving lesson with no violation, the driver has both hands on the wheel",
     "kind": "state", "window": 5, "threshold": 0.5, "min_consec": 1},
]


def load(taxonomy_path: str | None = None) -> list[dict]:
    """Default taxonomy, or an override JSON with the same list-of-dicts shape."""
    if taxonomy_path:
        return json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    return CLASSES


def class_ids(classes: list[dict]) -> list[str]:
    return [c["id"] for c in classes]


def violation_ids(classes: list[dict]) -> list[str]:
    return [c["id"] for c in classes if c["id"] != NEGATIVE_CLASS]


def caption_of(classes: list[dict]) -> dict[str, str]:
    return {c["id"]: c["caption"] for c in classes}
