"""Append-only JSONL shards that can be resumed: ids already on disk are never redone."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield the records of one JSONL file, skipping blank lines and a torn final line."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a write cut short by a crash; the id is simply not "done"
            if isinstance(record, dict):
                yield record


def shard_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []


def iter_records(directory: Path) -> Iterator[dict]:
    for path in shard_files(directory):
        yield from read_jsonl(path)


def done_ids(directory: Path, key: str = "id") -> set[str]:
    return {r[key] for r in iter_records(directory) if isinstance(r.get(key), str)}


class ShardWriter:
    """Writes records to `<directory>/<prefix>-00001.jsonl`, rolling over every `max_records`.

    Every record is flushed as it is written, so a crash loses at most the record in flight.
    Reopening continues the last shard (after repairing a missing final newline).
    """

    def __init__(self, directory: Path, prefix: str = "shard", max_records: int = 500):
        self.directory = Path(directory)
        self.prefix = prefix
        self.max_records = max_records
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.directory.glob(f"{prefix}-*.jsonl"))
        self._index = 1
        self._count = 0
        if existing:
            last = existing[-1]
            self._index = int(last.stem.rsplit("-", 1)[1])
            self._count = sum(1 for _ in read_jsonl(last))
            self._ensure_newline(last)

    @staticmethod
    def _ensure_newline(path: Path) -> None:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")

    def _path(self) -> Path:
        return self.directory / f"{self.prefix}-{self._index:05d}.jsonl"

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._count >= self.max_records:
                self._index += 1
                self._count = 0
            with self._path().open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
            self._count += 1
