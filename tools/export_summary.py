#!/usr/bin/env python
"""Цифры для страницы игры: python tools/export_summary.py пишет docs/data/summary.json.

Берёт модели из docs/models/index.json, контрольные матчи из results/final_*.json, точные проверки
и прямые матчи. Если контрольного матча нет, считает по последним проверкам силы при обучении.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from qubic.game import geometry  # noqa: E402

BOARDS = [(4, 4, 4), (3, 2, 3), (4, 2, 4), (5, 2, 5), (5, 2, 4), (7, 2, 5), (9, 2, 5), (15, 2, 5), (3, 3, 3), (4, 3, 4), (5, 3, 5)]
FINAL = {"4x4x4x4": "final_4x4x4x4.json", "4x4x4": "final_4x4x4_strong.json", "5x5x5": "final_5x5x5.json", "3x3x3": "final_3x3x3.json",
         "5x5": "final_5x5.json", "5x5k4": "final_5x5k4.json", "7x7k5": "final_7x7k5.json", "9x9k5": "final_9x9k5.json", "15x15k5": "final_15x15k5.json"}
# Цена игры, если её не считал наш решатель: 4×4×4 доказана в литературе, 5×5 доказана парной стратегией
KNOWN_VALUE = {"4x4x4": 1, "5x5": 0}
# Запуски на графике соперничества сетей на главной доске
RACE = ["n44", "c44", "c44n", "c44s", "x_b48", "x_gl"]
# Сеть из игры дообучалась по частям: запуск, с какой итерации считать, сколько итераций взять
CHAIN = {"x_gl": [("c44s", 0, 90), ("x_gs", 90, 25), ("x_gl", 115, 105)], "x_b48": [("x_b48", 0, 220)]}


def read_json(*parts):
    try:
        with open(os.path.join(ROOT, *parts)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_metrics(run):
    records = []
    for path in (os.path.join(ROOT, "runs", run, "metrics.jsonl"), os.path.join(ROOT, "results", "metrics", run + ".jsonl")):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        pass
            break
    return records


def percent(score):
    return round(100 * (score["win"] + score["draw"] / 2) / score["games"], 1)


def side(triple):
    total = sum(triple)
    return round(100 * (triple[0] + triple[1] / 2) / total, 1) if total else None


def scores_of(name, run, iteration):
    final = read_json("results", FINAL[name]) if name in FINAL else None
    if final and final.get("iteration") == iteration:
        table = final["scores"]
        sims, games = final.get("sims"), max(s["games"] for s in table.values())
    else:
        table = {}
        evals = [r["eval"] for r in read_metrics(run) if r.get("eval") and r["iter"] <= iteration][-3:]
        for result in evals:
            for opponent, s in result.items():
                if not isinstance(s, dict) or "games" not in s:
                    continue
                t = table.setdefault(opponent, {"games": 0, "win": 0, "draw": 0, "loss": 0, "as_first": [0, 0, 0], "as_second": [0, 0, 0]})
                for key in ("games", "win", "draw", "loss"):
                    t[key] += s[key]
                t["as_first"] = [a + b for a, b in zip(t["as_first"], s["as_first"])]
                t["as_second"] = [a + b for a, b in zip(t["as_second"], s["as_second"])]
        sims, games = None, max((s["games"] for s in table.values()), default=0)
    return {"games": games, "sims": sims,
            "vs": {o: {"score": percent(s), "first": side(s["as_first"]), "second": side(s["as_second"]), "loss": s["loss"]} for o, s in table.items()}}


def exact_of(run):
    check = read_json("runs", run, "perfect_check.json")
    if not check:
        return None
    decisions = sum(check[k]["decisions"] for k in ("первым", "вторым") if k in check)
    blunders = sum(check[k]["blunders"] for k in ("первым", "вторым") if k in check)
    every = check.get("every_position") or {}
    return {"value": check.get("first_move_value"), "decisions": decisions, "blunders": blunders,
            "positions": every.get("positions"), "complete": every.get("complete"), "mistakes": every.get("mistakes")}


def main():
    models = read_json("docs", "models", "index.json") or {}
    boards = []
    for n, d, k in BOARDS:
        geo = geometry(n, d, k)
        entry = {"name": geo.name, "n": n, "d": d, "k": geo.k, "cells": geo.cells, "lines": len(geo.lines),
                 "per_cell": round(len(geo.lines) * geo.k / geo.cells, 1), "value": KNOWN_VALUE.get(geo.name)}
        model = models.get(geo.name)
        if model:
            run = model.get("run")
            config = read_json("runs", run, "config.json") or {}
            entry.update(run=run, arch=config.get("arch"), channels=model["channels"], blocks=model["blocks"], iteration=model["iteration"],
                         **scores_of(geo.name, run, model["iteration"]))
            exact = exact_of(run)
            if exact:
                entry["exact"] = exact
                entry["value"] = exact["value"]
        boards.append(entry)
    race = []
    for run in RACE:
        config = read_json("runs", run, "config.json")
        if not config:
            continue
        points = []
        for part, start, count in CHAIN.get(run, [(run, 0, 10 ** 6)]):
            points += [[start + r["iter"], percent(r["eval"]["tsp"]), side(r["eval"]["tsp"]["as_second"])] for r in read_metrics(part)
                       if r.get("eval") and "tsp" in r["eval"] and r["iter"] <= count and (r["iter"] > 0 or start == 0)]
        race.append({"run": run, "arch": config.get("arch"), "channels": config.get("channels"), "blocks": config.get("blocks"), "points": points})
    matches = [m for m in (read_json("results", "matches.json") or []) if m["board"] == "4x4x4x4"]
    with open(os.path.join(ROOT, "docs", "data", "summary.json"), "w") as f:
        json.dump({"boards": boards, "race": race, "matches": matches}, f, ensure_ascii=False)
    print("досок", len(boards), "с сетью", sum("run" in b for b in boards), "запусков на графике", len(race), "матчей", len(matches))


if __name__ == "__main__":
    main()
