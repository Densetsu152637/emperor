"""Which records on disk count: the latest record per id, leaving out excluded pilots.

Pilots write into the full run's data directory, so their samples become training data and a
full run resumes from them instead of paying the teacher twice. Two rules keep that honest:

- A pilot whose teacher setup or guidance a later fix showed to be defective is listed in
  `<data-dir>/excluded-pilots.json` (`{"3": "why"}`). Its records stay on disk for the record
  but count for nothing: its tasks are generated again and its samples never reach the final
  set.
- Shards are append-only, so a task generated again or a sample judged again has several
  records. The last one written is the one that counts.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from emperor.data.shards import iter_records

EXCLUSIONS_FILE = "excluded-pilots.json"


def exclusions(directory: Path) -> dict[str, str]:
    """Excluded pilots ({pilot: reason}) for a data directory or one of its retry rounds."""
    for candidate in (directory, directory.parent):
        path = Path(candidate) / EXCLUSIONS_FILE
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()}
    return {}


def exclude_pilot(directory: Path, pilot: int | str, reason: str) -> dict[str, str]:
    path = Path(directory) / EXCLUSIONS_FILE
    data = exclusions(directory) if path.exists() else {}
    data[str(pilot)] = reason
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def pilot_of(record: dict) -> str | None:
    pilot = record.get("meta", {}).get("pilot", record.get("pilot"))
    return None if pilot is None else str(pilot)


def usable(record: dict, excluded: dict[str, str]) -> bool:
    return pilot_of(record) not in excluded


def current(directory: Path, excluded: dict[str, str] | None = None) -> dict[str, dict]:
    """The latest usable record per id in a shard directory."""
    excluded = exclusions(directory.parent) if excluded is None else excluded
    latest: dict[str, dict] = {}
    for record in iter_records(directory):
        if isinstance(record.get("id"), str) and usable(record, excluded):
            latest[record["id"]] = record
    return latest


def fingerprint(value) -> str:
    """A short, stable hash of JSON-able settings or text."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ids(records: Iterable[dict]) -> set[str]:
    return {r["id"] for r in records}
