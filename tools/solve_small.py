#!/usr/bin/env python
"""Точная проверка сети на маленькой доске: полным перебором узнаём, кто выигрывает в каждой позиции,
и считаем, сколько раз сеть выпустила выигрыш или ухудшила результат.

    python tools/solve_small.py runs/n3 --sims 100
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qubic.game import geometry
from qubic.mcts import Node, search
from qubic.net import Evaluator, build_net
from qubic.worker import CachedEvaluator


class Solver:
    """Полный перебор: +1 — ходящий выигрывает при идеальной игре, 0 — ничья, -1 — проигрывает."""

    def __init__(self, geo):
        self.geo = geo
        self.memo = {}

    def value(self, own, opp):
        key = (own, opp)
        if key in self.memo:
            return self.memo[key]
        geo = self.geo
        if geo.winning_cells(own, opp):
            result = 1
        elif own | opp == geo.full:
            result = 0
        else:
            threats = geo.winning_cells(opp, own)
            if len(threats) >= 2:
                result = -1
            elif threats:
                result = -self.value(opp, own | 1 << threats[0])
            else:
                result = -1
                moves = sorted(geo.empty_cells(own, opp), key=lambda c: -int(geo.degree[c]))
                for cell in moves:
                    result = max(result, -self.value(opp, own | 1 << cell))
                    if result == 1:
                        break
        self.memo[key] = result
        return result

    def after(self, own, opp, cell):
        """Оценка хода cell с точки зрения того, кто его делает."""
        mine = own | 1 << cell
        if self.geo.completes_line(mine, cell):
            return 1
        if mine | opp == self.geo.full:
            return 0
        return -self.value(opp, mine)


BATCH = 64


def net_moves(positions, evaluator, geo, sims, vcf_budget):
    """Ход сети в каждой позиции. Поиск считает их пачками: на видеокарте так намного быстрее."""
    cells, thinking = {}, []
    for own, opp in positions:
        root = Node(own, opp)
        root.analyse(geo)
        cell = geo.rules.find_vcf(own, opp, vcf_budget)
        if cell is None:
            cell = root.forced
        if cell is None:
            thinking.append(root)
        else:
            cells[own, opp] = cell
    for start in range(0, len(thinking), BATCH):
        group = thinking[start:start + BATCH]
        search(evaluator, group, sims, geo, vcf_budget=vcf_budget)
        for root in group:
            cells[root.own, root.opp] = root.moves[int(root.N.argmax())]
    return cells


def check_all_lines(solver, evaluator, geo, sims, vcf_budget, net_first):
    """Сеть ходит по своему, соперник перебирает все ответы: проверяем каждое решение сети."""
    seen = {(0, 0)}
    frontier, net_to_move = [(0, 0)], net_first
    decisions = blunders = net_wins = net_losses = draws = 0
    while frontier:
        chosen = net_moves(frontier, evaluator, geo, sims, vcf_budget) if net_to_move else {}
        following = []
        for own, opp in frontier:
            if net_to_move:
                decisions += 1
                cell = chosen[own, opp]
                blunders += solver.after(own, opp, cell) < solver.value(own, opp)
                cells = [cell]
            else:
                cells = geo.empty_cells(own, opp)
            for cell in cells:
                mine = own | 1 << cell
                if geo.completes_line(mine, cell):
                    if net_to_move:
                        net_wins += 1
                    else:
                        net_losses += 1
                elif mine | opp == geo.full:
                    draws += 1
                elif (opp, mine) not in seen:
                    seen.add((opp, mine))
                    following.append((opp, mine))
        frontier, net_to_move = following, not net_to_move
    return {"decisions": decisions, "blunders": blunders, "net_wins": net_wins, "net_losses": net_losses, "draws": draws}


def reachable_positions(geo, limit, rng):
    """Все незаконченные позиции, какие вообще могут встретиться, а если их слишком много, то случайная часть."""
    seen, stack = set(), [(0, 0)]
    while stack and len(seen) <= limit:
        own, opp = stack.pop()
        if (own, opp) in seen:
            continue
        seen.add((own, opp))
        for cell in geo.empty_cells(own, opp):
            mine = own | 1 << cell
            if not geo.completes_line(mine, cell) and mine | opp != geo.full:
                stack.append((opp, mine))
    if not stack:
        return seen, True
    seen = set()
    while len(seen) < limit:  # случайные партии: в них встречаются и нелепые позиции
        own, opp = 0, 0
        while True:
            seen.add((own, opp))
            free = geo.empty_cells(own, opp)
            cell = free[int(rng.integers(len(free)))]
            mine = own | 1 << cell
            if geo.completes_line(mine, cell) or mine | opp == geo.full:
                break
            own, opp = opp, mine
    return seen, False


def check_every_position(solver, evaluator, geo, sims, vcf_budget, limit, seed=1):
    """Самая строгая проверка: сеть ходит в любой возможной позиции, даже самой нелепой."""
    positions, complete = reachable_positions(geo, limit, np.random.default_rng(seed))
    chosen = net_moves(sorted(positions), evaluator, geo, sims, vcf_budget)
    names = {1: "выигранных", 0: "ничейных", -1: "проигранных"}
    decided = {name: {"positions": 0, "mistakes": 0} for name in names.values()}
    examples = []
    for (own, opp), cell in chosen.items():
        before = solver.value(own, opp)
        kind = decided[names[before]]
        kind["positions"] += 1
        if solver.after(own, opp, cell) < before:
            kind["mistakes"] += 1
            if len(examples) < 5:
                examples.append({"own": own, "opp": opp, "cell": cell, "before": before})
    return {"positions": len(chosen), "complete": complete, "mistakes": sum(v["mistakes"] for v in decided.values()),
            "by_value": decided, "examples": examples}


def opening_values(solver, geo, plies=2):
    """Что стоят начала из случайных ходов для того, кто ходит следующим: столько же получится в проверке."""
    counts = {1: 0, 0: 0, -1: 0}
    def walk(own, opp, left):
        if left == 0:
            counts[solver.value(own, opp)] += 1
            return
        for cell in geo.empty_cells(own, opp):
            walk(opp, own | 1 << cell, left - 1)
    walk(0, 0, plies)
    return {"win_for_next": counts[1], "draw": counts[0], "loss_for_next": counts[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--vcf-budget", type=int, default=100)
    p.add_argument("--positions", type=int, default=200000, help="сколько позиций проверить, если все не поместятся")
    args = p.parse_args()
    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    geo = geometry(cfg["n"], cfg["d"], cfg.get("k"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location=device, weights_only=True)
    net = build_net(cfg["arch"], geo, cfg["channels"], cfg["blocks"]).to(device).eval()
    net.load_state_dict(state["net"])
    evaluator = CachedEvaluator(Evaluator(net, device, geo))
    solver = Solver(geo)
    print(f"{geo.name}, итерация {state['iter']}. При идеальной игре первый ход: {solver.value(0, 0):+d}", flush=True)
    results = {}
    for net_first in (True, False):
        side = "первым" if net_first else "вторым"
        result = check_all_lines(solver, evaluator, geo, args.sims, args.vcf_budget, net_first)
        results[side] = result
        print(f"Сеть {side}, против всех ответов: решений {result['decisions']}, ошибок {result['blunders']};"
              f" концовок: побед сети {result['net_wins']}, поражений {result['net_losses']}, ничьих {result['draws']}", flush=True)
    openings = opening_values(solver, geo)
    print(f"Начала из двух случайных ходов: первый игрок выигрывает при идеальной игре в {openings['win_for_next']},"
          f" ничья в {openings['draw']}, проигрывает в {openings['loss_for_next']}", flush=True)
    every = check_every_position(solver, evaluator, geo, args.sims, args.vcf_budget, args.positions)
    print(f"Сеть ходит в {'любой возможной' if every['complete'] else 'случайной из возможных'} позиции: {every['positions']:,},"
          f" ошибок {every['mistakes']}. {every['by_value']}", flush=True)
    print(f"Позиций в таблице перебора: {len(solver.memo):,}", flush=True)
    with open(os.path.join(args.run, "perfect_check.json"), "w") as f:
        json.dump({"iter": state["iter"], "first_move_value": solver.value(0, 0), "openings": openings, "every_position": every, **results}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
