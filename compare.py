#!/usr/bin/env python
"""Прямые партии между обученными сетями: python compare.py runs/a runs/b@15 [runs/c ...] --games 200

Веса берутся из checkpoint.pt или, с --iter N либо runs/a@N, из nets/net_NNNN.pt. Настройки сети читаются из config.json.
Итоги дописываются в results/matches.json, оттуда их показывает дашборд.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from qubic.game import geometry
from qubic.net import Evaluator, build_net
from qubic.play import NetAgent, play_match
from qubic.worker import CachedEvaluator


MATCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "matches.json")


def save_match(record):
    try:
        with open(MATCHES) as f:
            records = json.load(f)
    except (OSError, ValueError):
        records = []
    records.append(record)
    with open(MATCHES + ".tmp", "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(MATCHES + ".tmp", MATCHES)


def load_net(spec, iteration, device):
    run, _, own_iter = spec.partition("@")  # runs/a@15 — своя итерация для этой сети
    iteration = int(own_iter) if own_iter else iteration
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    geo = geometry(cfg.get("n", 4), cfg.get("d", 3), cfg.get("k"))
    net = build_net(cfg.get("arch", "resnet"), geo, cfg["channels"], cfg["blocks"]).to(device).eval()
    if iteration:
        state = torch.load(os.path.join(run, "nets", f"net_{iteration:04d}.pt"), map_location=device, weights_only=True)
    else:
        checkpoint = torch.load(os.path.join(run, "checkpoint.pt"), map_location=device, weights_only=True)
        state, iteration = checkpoint["net"], checkpoint["iter"]
    net.load_state_dict(state)
    return net, geo, iteration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--iter", type=int, default=0)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--vcf-budget", type=int, default=400)
    p.add_argument("--root-safety", type=int, default=0, help="убирать в корне ходы, отдающие серию шахов; бюджет проверки")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    loaded = {run: load_net(run, args.iter, device) for run in args.runs}
    evaluators = {run: Evaluator(net, device, geo) for run, (net, geo, _) in loaded.items()}
    geo = next(iter(loaded.values()))[1]
    for i, a in enumerate(args.runs):
        for b in args.runs[i + 1:]:
            first = NetAgent(CachedEvaluator(evaluators[a]), args.sims, vcf_budget=args.vcf_budget, root_safety=args.root_safety)
            second = NetAgent(CachedEvaluator(evaluators[b]), args.sims, vcf_budget=args.vcf_budget, root_safety=args.root_safety)
            s, _ = play_match(first, second, args.games, rng, geo, opening_plies=2)
            rate = 100 * (s["win"] + s["draw"] / 2) / s["games"]
            print(f"{os.path.basename(a.rstrip('/'))} против {os.path.basename(b.rstrip('/'))}: {s['win']} побед, {s['draw']} ничьих, {s['loss']} поражений = {rate:.0f}%"
                  f" (первым {s['as_first']}, вторым {s['as_second']})", flush=True)
            save_match({"time": time.time(), "board": geo.name, "sims": args.sims, "games": s["games"],
                        "a": {"run": os.path.basename(a.partition("@")[0].rstrip("/")), "iter": loaded[a][2]},
                        "b": {"run": os.path.basename(b.partition("@")[0].rstrip("/")), "iter": loaded[b][2]},
                        "win": s["win"], "draw": s["draw"], "loss": s["loss"],
                        "as_first": s["as_first"], "as_second": s["as_second"]})


if __name__ == "__main__":
    main()
