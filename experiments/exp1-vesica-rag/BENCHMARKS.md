# Benchmarks

Throughput measurements recorded as we set up the slice. All numbers are on
**Apple M4 Pro / 48 GB RAM, macOS / MPS backend**.

## Contriever encoding (`facebook/contriever-msmarco`)

Mean-pool BERT-base, no quantization, fp32 compute, fp16 disk storage.

| batch_size | max_len | throughput (passages/sec) |
|---:|---:|---:|
| 128 | 256 |  68.7 |
| 256 | 256 |  65.8 |
| 512 | 256 |  62.8 |
| 256 | 128 | **139.9** ← chosen |
| 512 | 128 | 135.3 |

**Conclusion:** `max_len` dominates, batch size plateaus around 256. Chose
**batch=256, max_len=128** for the production encode. Median passage in
BeIR/hotpotqa is ~63 words (~100 tokens), so the 128-token truncation
affects only the long tail; both baseline and Vesica-RAG use the same
encoder with the same truncation, so the slice's primary metric (lift over
baseline) is invariant to this choice.

At 140 p/s, the full 5,233,329-passage corpus encodes in ~10.4 hours.
Sharded to 200,000 passages per `.npz` file at fp16 (~290 MB per shard,
~7.5 GB total).

## What's NOT in this number

- FAISS index build time (separate phase; small compared to encoding)
- 64-D random projection (effectively free; matrix multiply on already-encoded
  vectors)
- α and τ_v calibration (≤1 day each per PRECOMMIT.md)
- Eval inference via Ollama (separate budget item)
