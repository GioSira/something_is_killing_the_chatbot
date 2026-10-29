from typing import Any, Sequence, Iterator, Dict, Hashable, Mapping, List, Tuple
from typing import TypeVar

T = TypeVar("T")

def _hit_key(item: Any) -> str:
    return str(_read(item, "id", "_id"))


def _read(obj: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _batches(items: Sequence[Any], size: int = 16) -> Iterator[Sequence[Any]]:

    for start in range(0, len(items), size):
        yield items[start:start + size]


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[Any]], k: int = 60) -> List[Tuple[Any, float]]:
    """Fonde per id: lo stesso chunk trovato in più liste somma i contributi."""

    scores: Dict[str, float] = {}
    best: Dict[str, Any] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.id, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(best[i], score) for i, score in ordered]