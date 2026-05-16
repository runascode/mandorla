"""Train one cell of the Pilot 2 2×2.

  --arm  point | intersection
  --task and   | nonint
  --seed S

World seed is fixed (WORLD_SEED) so every cell sees the identical
world; only the bottleneck arm, the task target, and model init vary.
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

from src.losses import step_loss  # noqa: E402
from src.model import (  # noqa: E402
    ComposerModel,
    ModelConfig,
    assert_capacity_matched,
)
from src.probe import probe_report  # noqa: E402
from src.synthetic import BOS, QUERY, World, WorldConfig  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]
WORLD_SEED = 1337


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _corpus(world: World, max_len: int) -> torch.Tensor:
    rows = [p[:max_len] + [0] * (max_len - len(p)) for p in world.corpus()]
    return torch.tensor(rows, dtype=torch.long)


def _queries(world: World, task: str, device):
    sup = world.supervised_queries(task)
    prompt = torch.tensor(
        [[BOS, world.entity_tok(i), world.entity_tok(j), QUERY] for (i, j, _) in sup],
        dtype=torch.long, device=device,
    )
    tgt = torch.tensor(np.stack([a for (_, _, a) in sup]), dtype=torch.float32, device=device)
    return {"prompt": prompt, "target": tgt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["point", "intersection"], required=True)
    ap.add_argument("--task", choices=["and", "nonint"], required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=750)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.probe_every, args.log_every = 60, 60, 20

    device = _device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world = World(WorldConfig(seed=WORLD_SEED))
    mcfg = ModelConfig(vocab_size=world.vocab_size, n_props=world.cfg.n_props, seed=args.seed)
    model = ComposerModel(mcfg, args.arm).to(device)

    # Capacity audit (build the sibling arm just to compare counts).
    cap = assert_capacity_matched(
        ComposerModel(mcfg, "point"), ComposerModel(mcfg, "intersection")
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    corpus = _corpus(world, mcfg.max_len).to(device)
    n_para = corpus.size(0)
    q = _queries(world, args.task, device)
    n_q = q["prompt"].size(0)

    run = f"{args.arm}_{args.task}_seed{args.seed}"
    out = EXP_ROOT / "results"
    out.mkdir(exist_ok=True)
    log_f = (out / f"train_{run}.jsonl").open("w")
    rng = np.random.default_rng(args.seed)

    print(f"[{run}] dev={device} bottleneck_params point={cap['point_bottleneck_params']} "
          f"int={cap['int_bottleneck_params']} (int/point={cap['int_over_point']}) "
          f"corpus={n_para} sup_q={n_q} steps={args.steps}", flush=True)

    t0 = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        tok = corpus[torch.tensor(rng.integers(0, n_para, size=args.batch), device=device)]
        qi = torch.tensor(rng.integers(0, n_q, size=args.pair_batch), device=device)
        qb = {"prompt": q["prompt"][qi], "target": q["target"][qi]}
        loss, m = step_loss(model, tok, qb, args.lam)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            m.update({"step": step, "grad_norm": float(gn),
                      "elapsed_s": round(time.monotonic() - t0, 1)})
            log_f.write(json.dumps(m) + "\n")
            log_f.flush()
            print(f"[{run}] {step}/{args.steps} tot={m['loss_total']:.3f} "
                  f"clm={m['loss_clm']:.3f} ans={m['loss_answer']:.3f} "
                  f"abit={m['answer_bit_acc']:.3f}", flush=True)

        if step % args.probe_every == 0 or step == args.steps:
            rep = probe_report(model, world, args.task, device)
            rep["step"] = step
            (out / f"probe_{run}_step{step}.json").write_text(json.dumps(rep, indent=2))
            print(f"[{run}] PROBE {step}: comp-OOD F1@G={rep['comp_ood']['f1_at_g']:.4f} "
                  f"| seen-ctrl={rep['seen_entity_control']['f1_at_g']:.4f} "
                  f"| ood−seen={rep['ood_minus_seen']:+.4f}", flush=True)
            model.train()

    log_f.close()
    final = probe_report(model, world, args.task, device)
    final["capacity"] = cap
    (out / f"final_{run}.json").write_text(json.dumps(final, indent=2))
    print(f"[{run}] DONE comp-OOD F1@G={final['comp_ood']['f1_at_g']:.4f} "
          f"seen={final['seen_entity_control']['f1_at_g']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
