"""Held-out versus training overlap: exact duplicates and 3-gram similarity of the artifacts.

The held-out set is generated from a different seed range and a held-out vocabulary, and then
checked here. Tasks share templates, so similarity counts only distinctive word 3-grams (see
`OverlapIndex.closest`); a high score marks a near-copy, not shared template wording.
"""

from collections import defaultdict
from collections.abc import Iterable

from emperor.data.prompts import shingles

from consul.tasks import CritiqueTask

DEFAULT_THRESHOLD = 0.6


class OverlapIndex:
    def __init__(self, tasks: Iterable[CritiqueTask] = ()):
        self.ids: list[str] = []
        self.sizes: list[int] = []
        self.postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self.known_ids: set[str] = set()
        for task in tasks:
            self.add(task)

    def add(self, task: CritiqueTask) -> None:
        index = len(self.ids)
        grams = shingles(task.artifact)
        self.ids.append(task.id)
        self.sizes.append(len(grams))
        self.known_ids.add(task.id)
        for gram in grams:
            self.postings[gram].append(index)

    def closest(self, task: CritiqueTask) -> tuple[float, str | None]:
        """Highest similarity to any indexed task (1.0 for an exact id match).

        Only distinctive 3-grams count: those found in at most 2% of the indexed artifacts
        (and at least three), so shared template wording does not make everything look alike.
        The score is the share of the candidate's distinctive 3-grams found in one indexed task.
        """
        if task.id in self.known_ids:
            return 1.0, task.id
        cutoff = max(3, len(self.ids) // 50)
        rare = [g for g in shingles(task.artifact) if len(self.postings.get(g, ())) <= cutoff]
        if not rare:
            return 0.0, None
        shared: dict[int, int] = defaultdict(int)
        for gram in rare:
            for index in self.postings.get(gram, ()):
                shared[index] += 1
        if not shared:
            return 0.0, None
        index, count = max(shared.items(), key=lambda item: item[1])
        return count / len(rare), self.ids[index]


def overlap_report(
    train: Iterable[CritiqueTask], heldout: Iterable[CritiqueTask], threshold: float
) -> dict:
    index = OverlapIndex(train)
    worst, worst_pair, duplicates = 0.0, None, []
    count = 0
    for task in heldout:
        count += 1
        score, other = index.closest(task)
        if task.id in index.known_ids:
            duplicates.append(task.id)
        if score > worst:
            worst, worst_pair = score, [task.id, other]
    return {
        "train": len(index.ids),
        "heldout": count,
        "threshold": threshold,
        "max_overlap": round(worst, 4),
        "closest_pair": worst_pair,
        "exact_duplicates": duplicates,
        "ok": worst < threshold and not duplicates,
    }
