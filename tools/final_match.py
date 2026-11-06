#!/usr/bin/env python
"""Контрольный матч обученной сети против всех соперников и таблица результатов.

    python tools/final_match.py runs/main --games 500 --sims 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from qubic.game import geometry
from qubic.net import Evaluator, build_net


class Quiet:
    """Заменяет запись состояния обучения: здесь показывать нечего."""

    def set(self, **fields):
        pass


def main():
    from train import Pool, merge_scores, split_games

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--pure-sims", type=int, default=1000)
    p.add_argument("--vcf-budget", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--root-safety", type=int, default=None)
    p.add_argument("--opponents", default="osp,hsp,tsp,pure")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("n", 4), cfg.setdefault("d", 3)
    geo = geometry(cfg["n"], cfg["d"], cfg.get("k"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location=device, weights_only=True)
    net = build_net(cfg["arch"], geo, cfg["channels"], cfg["blocks"]).to(device).eval()
    net.load_state_dict(state["net"])

    cfg.update(eval_sims=args.sims, pure_sims=args.pure_sims, workers=args.workers, discount=1.0)
    if args.vcf_budget is not None:
        cfg["vcf_budget"] = args.vcf_budget
    if args.root_safety is not None:
        cfg["root_safety"] = args.root_safety
    pool = Pool(cfg, Quiet())
    rng = np.random.default_rng(2026)
    evaluators = {0: Evaluator(net, device, geo)}
    report = {"board": geo.name, "iteration": state["iter"], "sims": args.sims, "pure_sims": args.pure_sims, "scores": {}}
    try:
        for name in args.opponents.split(","):
            started = time.time()
            tasks = [{"kind": "match", "opponent": name, "games": games, "seed": int(rng.integers(2 ** 62))}
                     for games in split_games(args.games, args.workers)]
            score = merge_scores(pool.run(tasks, evaluators))[name]
            report["scores"][name] = score
            rate = 100 * (score["win"] + score["draw"] / 2) / score["games"]
            print(f"{geo.name} итерация {state['iter']}, {args.sims} симуляций против {name}: {score['win']}-{score['draw']}-{score['loss']} = {rate:.1f}%"
                  f" (первым {score['as_first']}, вторым {score['as_second']}), {time.time() - started:.0f} с", flush=True)
    finally:
        pool.close()
    out = args.out or os.path.join(os.path.dirname(__file__), "..", "results", f"final_{geo.name}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
