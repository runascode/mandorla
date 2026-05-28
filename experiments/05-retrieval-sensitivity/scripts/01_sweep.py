"""Retrieval-sensitivity sweep — the core figure for the standalone
reader-saturation paper. Reuses Exp 01's FAISS index, contriever
encoder, Ollama reader, and eval, bit-for-bit. Resumable JSONL.

Four retrieval conditions (same reader/prompt/budget=25); design locked
in ../DESIGN.md. Run:

  uv run python scripts/01_sweep.py            # all conditions, 500 q
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

EXP01 = Path(__file__).resolve().parents[2] / "01-vesica-rag"
sys.path.insert(0, str(EXP01))

from src.data import TitleIndex, load_hotpotqa          # noqa: E402
from src.eval import exact_match, f1_score              # noqa: E402
from src.generate import OllamaGenerator, RetrievedChunk, build_prompt  # noqa: E402
from src.index_io import QueryEncoder, load_faiss_retriever  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "sweep.jsonl"
N = 500
BUDGET = 25
SEED = 1337
CONDITIONS = ("oracle", "dense", "gold_removed", "random")


def _done() -> set[tuple[str, str]]:
    if not RAW.exists():
        return set()
    out = set()
    for line in RAW.read_text().splitlines():
        try:
            r = json.loads(line)
            out.add((r["id"], r["condition"]))
        except Exception:
            pass
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    questions = list(load_hotpotqa("validation"))
    idx = rng.choice(len(questions), size=min(N, len(questions)), replace=False)
    sample = [questions[i] for i in sorted(idx)]

    print("Loading FAISS retriever (~16 GB)…", flush=True)
    retr = load_faiss_retriever()
    titles = TitleIndex()
    enc = QueryEncoder()
    gen = OllamaGenerator()
    all_ids = retr.chunk_ids
    done = _done()

    def chunks_for(cond: str, q) -> list[str]:
        gold_ids = [titles.title_to_id.get(t) for t in q.supporting_titles]
        gold_ids = [g for g in gold_ids if g]
        qv = enc.encode(q.question)
        dense = [s.chunk_id for s in retr.top_k(qv, 200)]
        if cond == "dense":
            return dense[:BUDGET]
        if cond == "oracle":
            picked = list(dict.fromkeys(gold_ids))
            for c in dense:
                if len(picked) >= BUDGET:
                    break
                if c not in picked:
                    picked.append(c)
            return picked[:BUDGET]
        if cond == "gold_removed":
            gold_titles = set(q.supporting_titles)
            out = []
            for c in dense:
                try:
                    if titles.passage(c)["title"] in gold_titles:
                        continue
                except KeyError:
                    pass
                out.append(c)
                if len(out) >= BUDGET:
                    break
            return out[:BUDGET]
        # random
        ri = rng.choice(len(all_ids), size=BUDGET, replace=False)
        return [all_ids[i] for i in ri]

    t0 = time.monotonic()
    n_target = len(sample) * len(CONDITIONS)
    n_have = len(done)
    with RAW.open("a") as f:
        for cond in CONDITIONS:
            for qi, q in enumerate(sample):
                if (q.id, cond) in done:
                    continue
                cids = chunks_for(cond, q)
                rcs = []
                gold_titles = set(q.supporting_titles)
                got_titles = set()
                for c in cids:
                    try:
                        p = titles.passage(c)
                    except KeyError:
                        continue
                    rcs.append(RetrievedChunk(chunk_id=c, title=p["title"], text=p["text"]))
                    got_titles.add(p["title"])
                res = gen.answer_question(q.question, rcs)
                rec = {
                    "id": q.id, "condition": cond,
                    "f1": f1_score(res.answer, q.answer),
                    "em": exact_match(res.answer, q.answer),
                    "pair_in_context": int(gold_titles.issubset(got_titles)),
                    "n_chunks": len(rcs),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_have += 1
                if n_have % 25 == 0:
                    dt = time.monotonic() - t0
                    qps = (n_have - len(done)) / dt if dt > 0 else 0
                    eta = (n_target - n_have) / qps / 60 if qps > 0 else float("nan")
                    print(f"  {cond} {n_have}/{n_target} ({qps:.2f} q/s, eta {eta:.0f} min)",
                          flush=True)
    print("done. processed sweep", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
