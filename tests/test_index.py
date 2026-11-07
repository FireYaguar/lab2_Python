import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import DocMeta, Posting, build_index
from findex.search import ENGINES, search
from findex.store import load, save

DOCS = [
    Document(0, "a.txt", "A", "python is fast python"),
    Document(1, "b.txt", "B", "java is verbose"),
    Document(2, "c.txt", "C", "python and java"),
    Document(3, "d.txt", "D", "rust"),
]


@pytest.fixture
def index():
    return build_index(iter(DOCS), positions=True)


def test_postings_sorted_with_tf_and_positions(index) -> None:
    assert index.postings["python"] == [Posting(0, 2, (0, 3)), Posting(2, 1, (0,))]
    for plist in index.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)
    assert index.doc_lengths == {0: 4, 1: 3, 2: 3, 3: 1}
    assert index.doc_meta[3] == DocMeta(3, "d.txt", "D")


def test_records_are_hashable_and_frozen() -> None:
    assert len({Posting(1, 2), Posting(1, 2), DocMeta(1, "p", "t")}) == 2
    with pytest.raises(AttributeError):
        Posting(1, 2).tf = 5  # type: ignore[misc]


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", [0, 2]),
        ("python java", [2]),
        ("python OR rust", [0, 2, 3]),
        ("python NOT java", [0]),
        ("NOT python", [1, 3]),
        ("Python, IS!", [0]),
        ("missing", []),
    ],
)
def test_search(index, engine, query, expected) -> None:
    assert search(index, query, engine) == expected


def test_merge_matches_set_on_random_lists() -> None:
    rng = random.Random(0)
    for _ in range(200):
        a = sorted(rng.sample(range(100), rng.randint(0, 30)))
        b = sorted(rng.sample(range(100), rng.randint(0, 30)))
        for m, s in zip(ENGINES["merge"], ENGINES["set"]):
            assert m(a, b) == s(a, b)


@pytest.mark.parametrize("name", ["index.bin", "index.json"])
def test_save_load_round_trip(index, tmp_path: Path, name: str) -> None:
    save(index, tmp_path / name)
    assert load(tmp_path / name) == index
