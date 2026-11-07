"""Inverted index: term -> postings sorted by doc_id. Built in ONE pass over the lazy corpus.

    python -m findex.index data/ --out index.bin [--positions] [--limit N]

The output format is chosen by suffix: *.json -> JSON, anything else -> pickle (see store.py).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # term frequency in this document
    positions: tuple[int, ...] = ()  # token ordinals (0, 1, 2, ...) — only with --positions


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


@dataclass(slots=True)
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict[str, list[Posting]])
    doc_lengths: dict[int, int] = field(default_factory=dict[int, int])  # tokens per doc (BM25)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict[int, DocMeta])
    has_positions: bool = False

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)


def build_index(docs: Iterable[Document], *, positions: bool = False) -> InvertedIndex:
    """One streaming pass. Per document we keep a Counter (or term -> positions),
    flush it into the postings and drop it. doc_ids arrive increasing, so every
    postings list is sorted by doc_id without an explicit sort."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    index = InvertedIndex(has_positions=positions)
    for doc in docs:
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            n = 0
            for n, tok in enumerate(tokenize(doc.text), 1):
                pos[tok].append(n - 1)
            for term, ps in pos.items():
                postings[term].append(Posting(doc.doc_id, len(ps), tuple(ps)))
            length = n
        else:
            counts = Counter(tokenize(doc.text))
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))
            length = counts.total()
        index.doc_lengths[doc.doc_id] = length
        index.doc_meta[doc.doc_id] = DocMeta(doc.doc_id, doc.path, doc.title)
    index.postings = dict(postings)  # plain dict: no accidental inserts on lookup of a missing term
    return index


def main(argv: list[str] | None = None) -> None:
    from findex.stats import measure
    from findex.store import save

    ap = argparse.ArgumentParser(prog="python -m findex.index", description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", action="store_true", help="store token positions (phrases)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    index, t_build, peak = measure(
        lambda: build_index(islice(iter_documents(args.root), args.limit), positions=args.positions)
    )
    _, t_save, _ = measure(lambda: save(index, args.out))
    n_postings = sum(len(p) for p in index.postings.values())
    print(f"documents:   {index.num_docs:,}")
    print(f"terms:       {len(index.postings):,}")
    print(f"postings:    {n_postings:,}")
    print(f"build:       {t_build:.2f} s, peak memory {peak / 2**20:.1f} MiB (tracemalloc)")
    print(f"save:        {t_save:.2f} s -> {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    # Run via the real module, not __main__: otherwise pickle records the classes as
    # "__main__.Posting" and no other program can load the file.
    from findex.index import main as _main

    _main()
