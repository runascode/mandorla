"""Train one condition (baseline | curriculum) on the synthetic world.

Both conditions share the world (fixed world seed → identical corpus
and held-out split); only the loss differs and only the model init
varies with --seed (for multi-seed runs). Per-step stability metrics
go to a JSONL — Q1 of the pilot is "do these losses train without
collapsing", so the instrumentation is the deliverable, not a print.

Usage:
  uv run python scripts/02_train.py --condition curriculum --seed 1337
  uv run python scripts/02_train.py --condition baseline   --seed 1337
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import baseline_loss, curriculum_loss  # noqa: E402
from src.model import ModelConfig, TinyTransformer, VesicaHeads  # noqa: E402
from src.probe import probe_report  # noqa: E402
from src.synthetic import PAD, World, WorldConfig  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]
WORLD_SEED = 1337  # fixed: both conditions, every model-seed, see the same world


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _padded_corpus(world: World, max_len: int) -> torch.Tensor:
    rows = []
    for para in world.corpus():
        r = para[:max_len] + [PAD] * (max_len - len(para))
        rows.append(r)
    return torch.tensor(rows, dtype=torch.long)


def _pair_tensors(world: World, device: torch.device):
    sup = world.supervised_pairs()
    ei = torch.tensor([world.entity_tok(i) for (i, j, _) in sup], device=device)
    ej = torch.tensor([world.entity_tok(j) for (i, j, _) in sup], device=device)
    shared = torch.tensor(np.stack([s for (_, _, s) in sup]), dtype=torch.float32, device=device)
    attr_i = torch.tensor(np.stack([world.A[i] for (i, _, _) in sup]), dtype=torch.float32, device=device)
    attr_j = torch.tensor(np.stack([world.A[j] for (_, j, _) in sup]), dtype=torch.float32, device=device)
    return ei, ej, shared, attr_i, attr_j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "curriculum"], required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--lambda-p", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the loop executes")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.probe_every, args.log_every = 50, 50, 10

    device = _device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world = World(WorldConfig(seed=WORLD_SEED))
    mcfg = ModelConfig(vocab_size=world.vocab_size, seed=args.seed)
    model = TinyTransformer(mcfg).to(device)
    heads = VesicaHeads(mcfg.box_dim, world.cfg.n_props).to(device)

    params = list(model.parameters())
    if args.condition == "curriculum":
        params += list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    corpus = _padded_corpus(world, mcfg.max_len).to(device)
    n_para = corpus.size(0)
    ei, ej, shared, attr_i, attr_j = _pair_tensors(world, device)
    n_sup = ei.size(0)

    run = f"{args.condition}_seed{args.seed}"
    out_dir = EXP_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"train_{run}.jsonl"
    log_f = log_path.open("w")
    rng = np.random.default_rng(args.seed)

    print(f"[{run}] device={device} params={model.n_params():,} "
          f"corpus={n_para} sup_pairs={n_sup} steps={args.steps}", flush=True)

    t0 = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        bi = rng.integers(0, n_para, size=args.batch)
        tok = corpus[torch.tensor(bi, device=device)]

        if args.condition == "curriculum":
            pj = rng.integers(0, n_sup, size=args.pair_batch)
            pj = torch.tensor(pj, device=device)
            pair_batch = {
                "ei": ei[pj], "ej": ej[pj], "shared": shared[pj],
                "attr_i": attr_i[pj], "attr_j": attr_j[pj],
            }
            loss, metrics = curriculum_loss(
                model, heads, tok, pair_batch, args.lambda_v, args.lambda_p
            )
        else:
            loss, metrics = baseline_loss(model, tok)

        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            metrics.update({"step": step, "grad_norm": float(gnorm),
                            "elapsed_s": round(time.monotonic() - t0, 1)})
            log_f.write(json.dumps(metrics) + "\n")
            log_f.flush()
            print(f"[{run}] step {step}/{args.steps} "
                  f"loss={metrics['loss_total']:.4f} "
                  f"clm={metrics.get('loss_clm', float('nan')):.4f}"
                  + (f" ves={metrics['loss_vesica']:.4f} par={metrics['loss_parent']:.4f}"
                     f" ilv={metrics['inter_logvol_mean']:.2f}"
                     if args.condition == "curriculum" else ""),
                  flush=True)

        if step % args.probe_every == 0 or step == args.steps:
            rep = probe_report(model, world, device)
            rep["step"] = step
            (out_dir / f"probe_{run}_step{step}.json").write_text(json.dumps(rep, indent=2))
            print(f"[{run}] PROBE step {step}: "
                  f"heldout F1@G={rep['heldout']['f1_at_g']:.4f} "
                  f"exact={rep['heldout']['exact_set_acc']:.4f} | "
                  f"seen-ctrl F1@G={rep['seen_control']['f1_at_g']:.4f}",
                  flush=True)
            model.train()

    log_f.close()
    final = probe_report(model, world, device)
    (out_dir / f"final_{run}.json").write_text(json.dumps(final, indent=2))
    print(f"[{run}] DONE final heldout F1@G={final['heldout']['f1_at_g']:.4f} "
          f"exact={final['heldout']['exact_set_acc']:.4f} "
          f"seen-ctrl F1@G={final['seen_control']['f1_at_g']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
