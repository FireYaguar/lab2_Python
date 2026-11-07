"""Lab 2 measurements -> markdown tables for the README.

uv run python scripts/lab02_measure.py data/
"""

from __future__ import annotations

import pickle
import statistics
import sys
import tempfile
import time
import timeit
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import InvertedIndex, build_index
from findex.search import ENGINES, doc_ids
from findex.store import load, save
from findex.tokenize import tokenize

MiB = 2**20


def median_time(fn: Callable[[], object], runs: int = 3) -> float:
    ts: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------- 1. merge vs set ----------
def bench_engines(index: InvertedIndex) -> None:
    by_df = sorted(index.postings, key=lambda t: len(index.postings[t]))
    common, rare = by_df[-2:], by_df[:2]
    pairs = [
        ("2 most common", common),
        ("2 rarest", rare),
        ("common + rare", [common[-1], rare[0]]),
    ]
    print("| Terms | df | Op | merge (µs) | set (µs) |\n|---|---|---|---|---|")
    for label, (t1, t2) in pairs:
        a, b = doc_ids(index, t1), doc_ids(index, t2)
        for op_i, op in enumerate(["AND", "OR"]):
            res = []
            for eng in ("merge", "set"):
                f = ENGINES[eng][op_i]
                n, total = timeit.Timer(lambda f=f, a=a, b=b: f(a, b)).autorange()
                res.append(total / n * 1e6)
            print(
                f"| {label}: `{t1}`, `{t2}` | {len(a)} / {len(b)} | {op} "
                f"| {res[0]:.1f} | {res[1]:.1f} |"
            )


# ---------- 2. formats ----------
def bench_formats(root: Path) -> None:
    print("\n| Format | Positions | File size | save (s) | load (s) |\n|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for positions in (False, True):
            index = build_index(iter_documents(root), positions=positions)
            for name in ("index.bin", "index.json"):
                p = Path(tmp) / name
                t_save = median_time(lambda i=index, p=p: save(i, p))
                t_load = median_time(lambda p=p: load(p))
                fmt = "pickle" if name.endswith(".bin") else "json"
                print(
                    f"| {fmt} | {'yes' if positions else 'no'} | {p.stat().st_size / MiB:.1f} MiB "
                    f"| {t_save:.2f} | {t_load:.2f} |"
                )


# ---------- 3. three postings representations ----------
@dataclass(frozen=True)
class PlainPosting:  # regular dataclass: every instance carries a __dict__
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


def build_objects(root: Path, cls: type) -> dict[str, list[object]]:
    postings: defaultdict[str, list[object]] = defaultdict(list)
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            postings[term].append(cls(doc.doc_id, tf))
    return dict(postings)


def build_arrays(root: Path) -> dict[str, tuple[array[int], array[int]]]:
    postings: dict[str, tuple[array[int], array[int]]] = {}
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            pair = postings.get(term)
            if pair is None:
                pair = postings[term] = (array("I"), array("I"))
            pair[0].append(doc.doc_id)
            pair[1].append(tf)
    return postings


def memory_study(root: Path) -> None:
    variants: list[tuple[str, Callable[[], object]]] = [
        ("`list[Posting]`, plain `@dataclass`", lambda: build_objects(root, PlainPosting)),
        ("`list[Posting]`, `slots=True`", lambda: build_objects(root, SlotPosting)),
        ("`array('I')` pairs (doc_ids, tfs)", lambda: build_arrays(root)),
    ]
    print(
        "\n| Postings representation | Peak (build) | Retained index | File (pickle) | Load (s) |"
    )
    print("|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for label, build in variants:
            tracemalloc.start()
            obj = build()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            p = Path(tmp) / "x.pkl"
            p.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            t_load = median_time(lambda p=p: pickle.loads(p.read_bytes()))
            print(
                f"| {label} | {peak / MiB:.1f} MiB | {current / MiB:.1f} MiB "
                f"| {p.stat().st_size / MiB:.1f} MiB | {t_load:.2f} |"
            )
            del obj


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    idx = build_index(iter_documents(root))
    n = sum(len(v) for v in idx.postings.values())
    print(f"corpus: {idx.num_docs} docs, {len(idx.postings):,} terms, {n:,} postings\n")
    bench_engines(idx)
    del idx
    bench_formats(root)
    memory_study(root)
