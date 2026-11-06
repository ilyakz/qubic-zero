#!/usr/bin/env python
"""Готовит сеть для игры в браузере: docs/models/<доска>.json и .bin (float32).

    python tools/export_web.py runs/main
    python tools/export_web.py runs/main --fixtures tests/web_fixtures.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qubic.game import geometry
from qubic.net import Evaluator, build_net

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--iter", type=int, default=0, help="взять снимок nets/net_NNNN.pt вместо последней итерации")
    p.add_argument("--models-dir", default=os.path.join(ROOT, "docs", "models"), help="куда записать модель")
    p.add_argument("--fixtures", default=None, help="записать позиции и ответы сети для проверки версии на JavaScript")
    args = p.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    if cfg["arch"] not in ("linenet", "conv4d", "conv4d_lines"):
        sys.exit("В браузере работают сети linenet, conv4d и conv4d_lines")
    geo = geometry(cfg["n"], cfg["d"], cfg.get("k"))
    if args.iter:
        state, iteration = torch.load(os.path.join(args.run, "nets", f"net_{args.iter:04d}.pt"), map_location="cpu", weights_only=True), args.iter
    else:
        checkpoint = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu", weights_only=True)
        state, iteration = checkpoint["net"], checkpoint["iter"]
    net = build_net(cfg["arch"], geo, cfg["channels"], cfg["blocks"]).eval()
    net.load_state_dict(state)

    # таблицу линий браузер строит сам, счётчик шагов нормировки при игре не нужен
    state = {name: tensor for name, tensor in state.items() if name != "features.incidence" and not name.endswith("num_batches_tracked")}
    params = [{"name": name, "size": int(tensor.numel())} for name, tensor in state.items()]
    os.makedirs(args.models_dir, exist_ok=True)
    base = os.path.join(args.models_dir, geo.name)
    np.concatenate([tensor.numpy().astype("<f4").ravel() for tensor in state.values()]).tofile(base + ".bin")
    with open(base + ".json", "w") as f:
        json.dump({"arch": cfg["arch"], "n": cfg["n"], "d": cfg["d"], "k": geo.k, "channels": cfg["channels"], "blocks": cfg["blocks"],
                   "iteration": iteration, "params": params}, f)
    print(f"{geo.name}: итерация {iteration}, {sum(q['size'] for q in params):,} весов, {os.path.getsize(base + '.bin') / 1e6:.1f} МБ")
    index_path = os.path.join(args.models_dir, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    index[geo.name] = {"arch": cfg["arch"], "iteration": iteration, "channels": cfg["channels"], "blocks": cfg["blocks"], "run": os.path.basename(args.run.rstrip("/"))}
    with open(index_path, "w") as f:
        json.dump(index, f)

    if args.fixtures:
        rng = np.random.default_rng(1)
        cases = []
        evaluator = Evaluator(net, torch.device("cpu"), geo)
        while len(cases) < 40:
            moves, own, opp = [], 0, 0
            for _ in range(int(rng.integers(0, geo.cells // 2))):
                free = geo.empty_cells(own, opp)
                cell = free[int(rng.integers(len(free)))]
                if geo.completes_line(own | 1 << cell, cell):
                    break
                moves.append(cell)
                own, opp = opp, own | 1 << cell
            priors, values = evaluator([own], [opp])
            cases.append({"moves": moves, "priors": [round(float(v), 6) for v in priors[0]], "value": round(float(values[0]), 6),
                          "wins": sorted(geo.winning_cells(own, opp)), "threats": sorted(geo.winning_cells(opp, own)),
                          "vcf": None if geo.winning_cells(own, opp) else geo.rules.find_vcf(own, opp, budget=3000, depth=64) is not None})
        with open(args.fixtures, "w") as f:
            json.dump({"n": cfg["n"], "d": cfg["d"], "k": geo.k, "lines": len(geo.lines), "cases": cases}, f)
        print("проверочных позиций:", len(cases))


if __name__ == "__main__":
    main()
