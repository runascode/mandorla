"""Train one condition on the R1 2-hop relational world.

  --condition baseline | generic_aux | curriculum

All conditions share the world (fixed WORLD_SEED → identical corpus and
held-out-entity split); only the auxiliary loss and the model init
(--seed) vary. Per-step stability metrics → results/train_*.jsonl
(Q1). Probe (comp-OOD + seen-entity control) → results/*.json (Q2).
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

from src.losses import baseline_loss, curriculum_loss, generic_aux_loss  # noqa: E402
from src.model import (  # noqa: E402
    GenericAuxHead,
    ModelConfig,
    TinyTransformer,
    VesicaHeads,
)
from src.probe import probe_report  # noqa: E402
from src.synthetic import BOS, QUERY, World, WorldConfig  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]
WORLD_SEED = 1337


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _padded_corpus(world: World, max_len: int) -> torch.Tensor:
    rows = [p[:max_len] + [0] * (max_len - len(p)) for p in world.corpus()]
    return torch.tensor(rows, dtype=torch.long)


def _query_tensors(world: World, device: torch.device) -> dict:
    sup = world.supervised_queries()                 # trained entities only
    ei = np.array([i for (i, j, _) in sup])
    ej = np.array([j for (i, j, _) in sup])
    prompt = torch.tensor(
        [[BOS, world.entity_tok(int(i)), world.entity_tok(int(j)), QUERY]
         for i, j in zip(ei, ej)],
        dtype=torch.long, device=device,
    )
    ans = torch.tensor(np.stack([a for (_, _, a) in sup]), dtype=torch.float32, device=device)
    attr_i = torch.tensor(np.stack([world.A[i] for i in ei]), dtype=torch.float32, device=device)
    attr_pi = torch.tensor(np.stack([world.A[world.pi[i]] for i in ei]), dtype=torch.float32, device=device)
    attr_j = torch.tensor(np.stack([world.A[j] for j in ej]), dtype=torch.float32, device=device)
    return {"prompt": prompt, "answer": ans, "attr_i": attr_i,
            "attr_pi": attr_pi, "attr_j": attr_j}


def _slice(q: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in q.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "generic_aux", "curriculum"], required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--lambda-p", type=float, default=1.0)
    ap.add_argument("--lambda-aux", type=float, default=1.0)
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
    mcfg = ModelConfig(vocab_size=world.vocab_size, seed=args.seed)
    model = TinyTransformer(mcfg).to(device)
    heads = VesicaHeads(mcfg.box_dim, world.cfg.n_props).to(device)
    ghead = GenericAuxHead(mcfg.d_model, world.cfg.n_props).to(device)

    params = list(model.parameters())
    extra = 0
    if args.condition == "curriculum":
        params += list(heads.parameters())
        extra = sum(p.numel() for p in heads.parameters()) + sum(
            p.numel() for p in model.box_head.parameters())
    elif args.condition == "generic_aux":
        params += list(ghead.parameters())
        extra = sum(p.numel() for p in ghead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    corpus = _padded_corpus(world, mcfg.max_len).to(device)
    n_para = corpus.size(0)
    q = _query_tensors(world, device)
    n_q = q["prompt"].size(0)

    run = f"{args.condition}_seed{args.seed}"
    out_dir = EXP_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    log_f = (out_dir / f"train_{run}.jsonl").open("w")
    rng = np.random.default_rng(args.seed)

    print(f"[{run}] device={device} base_params={model.n_params():,} "
          f"aux_extra_params={extra:,} corpus={n_para} sup_q={n_q} "
          f"heldout_entities={len(world.heldout_entities)} steps={args.steps}",
          flush=True)

    t0 = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        tok = corpus[torch.tensor(rng.integers(0, n_para, size=args.batch), device=device)]
        if args.condition == "baseline":
            loss, m = baseline_loss(model, tok)
        else:
            qi = torch.tensor(rng.integers(0, n_q, size=args.pair_batch), device=device)
            qb = _slice(q, qi)
            if args.condition == "curriculum":
                loss, m = curriculum_loss(model, heads, tok, qb, args.lambda_v, args.lambda_p)
            else:
                loss, m = generic_aux_loss(model, ghead, tok, qb, args.lambda_aux)

        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            m.update({"step": step, "grad_norm": float(gnorm),
                      "elapsed_s": round(time.monotonic() - t0, 1)})
            log_f.write(json.dumps(m) + "\n")
            log_f.flush()
            extra_s = ""
            if args.condition == "curriculum":
                extra_s = (f" ves={m['loss_vesica']:.3f} par={m['loss_parent']:.3f}"
                           f" ilv={m['inter_logvol_mean']:.2f}")
            elif args.condition == "generic_aux":
                extra_s = f" aux={m['loss_generic_aux']:.3f}"
            print(f"[{run}] {step}/{args.steps} loss={m['loss_total']:.3f} "
                  f"clm={m['loss_clm']:.3f}{extra_s}", flush=True)

        if step % args.probe_every == 0 or step == args.steps:
            rep = probe_report(model, world, device)
            rep["step"] = step
            (out_dir / f"probe_{run}_step{step}.json").write_text(json.dumps(rep, indent=2))
            print(f"[{run}] PROBE {step}: comp-OOD F1@G={rep['comp_ood']['f1_at_g']:.4f} "
                  f"exact={rep['comp_ood']['exact_set_acc']:.4f} | "
                  f"seen-ctrl F1@G={rep['seen_entity_control']['f1_at_g']:.4f} | "
                  f"ood−seen={rep['ood_minus_seen']:+.4f}", flush=True)
            model.train()

    log_f.close()
    final = probe_report(model, world, device)
    (out_dir / f"final_{run}.json").write_text(json.dumps(final, indent=2))
    print(f"[{run}] DONE comp-OOD F1@G={final['comp_ood']['f1_at_g']:.4f} "
          f"seen-ctrl={final['seen_entity_control']['f1_at_g']:.4f} "
          f"ood−seen={final['ood_minus_seen']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
