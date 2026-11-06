"""Игровые процессы. Здесь только дерево поиска, а сеть считает главный процесс:
так позиции всех партий попадают на видеокарту одной большой пачкой.

В этом модуле нельзя импортировать torch: игровым процессам он не нужен.
"""
import os
import pickle
import time
import traceback

import numpy as np

from .bots import BOTS
from .game import geometry
from .play import NetAgent, play_match, replay_arrays, self_play

REQUEST, PROGRESS, RESULT = b"I", b"P", b"R"


class CachedEvaluator:
    """Запоминает ответы сети: одна и та же расстановка камней получается разным порядком ходов.

    Пока веса сети не меняются, ответ для позиции тоже не меняется, поэтому объект живёт одну задачу.
    """

    def __init__(self, infer, limit=400000):
        self.infer = infer
        self.limit = limit
        self.cache = {}
        self.requests = 0
        self.computed = 0

    def __call__(self, owns, opps):
        cache = self.cache
        keys = list(zip(owns, opps))
        self.requests += len(keys)
        if len(cache) > self.limit:
            cache.clear()
        missing = {}
        for key in keys:
            if key not in cache:
                missing.setdefault(key, len(missing))
        if missing:
            probs, values = self.infer([k[0] for k in missing], [k[1] for k in missing])
            self.computed += len(missing)
            for key, i in missing.items():
                cache[key] = (probs[i], float(values[i]))
        answers = [cache[key] for key in keys]
        return [a[0] for a in answers], [a[1] for a in answers]


def remote_net(conn, net_id, geo):
    """Функция оценки, которая отправляет позиции в главный процесс и ждёт ответа."""
    header = REQUEST + bytes([net_id])

    def infer(owns, opps):
        n = len(owns)
        conn.send_bytes(header + geo.pack(owns + opps).tobytes())
        reply = conn.recv_bytes()
        probs = np.frombuffer(reply, np.float32, n * geo.cells).reshape(n, geo.cells)
        values = np.frombuffer(reply, np.float32, n, offset=n * geo.cells * 4)
        return probs, values

    return infer


def uniform_net(geo):
    """Сеть без знаний: все ходы равновероятны, оценка любой позиции 0."""
    def infer(owns, opps):
        n = len(owns)
        return np.full((n, geo.cells), 1.0 / geo.cells, dtype=np.float32), np.zeros(n, dtype=np.float32)
    return infer


def run_task(task, cfg, conn, geo):
    rng = np.random.default_rng(task["seed"])
    reported = [0]

    def on_progress(done, total):
        conn.send_bytes(PROGRESS + (done - reported[0]).to_bytes(4, "little"))
        reported[0] = done

    evaluator = CachedEvaluator(remote_net(conn, 0, geo))
    if task["kind"] == "selfplay":
        started = time.time()
        league = [(BOTS[name], share) for name, share in cfg.get("league_shares", {}).items()]
        games, sims = self_play(evaluator, task["games"], cfg["parallel"], cfg["sims"], rng, geo,
                                cfg["temperature_plies"], cfg["c_puct"], cfg["fpu"],
                                cfg["noise_alpha"], cfg["noise_eps"], on_progress, cfg["early_finish"], cfg["vcf_budget"],
                                cfg["random_plies"], cfg["discount"], cfg.get("root_safety", 0), league)
        against_bots = [g for g in games if g.bot is not None]
        by_itself = [g for g in games if g.bot is None]
        return {"arrays": replay_arrays(games, geo, cfg["legacy_labels"], cfg["discount"]),
                "winners": [g.winner for g in by_itself], "lengths": [len(g.moves) for g in by_itself],
                "bot_games": len(against_bots), "bot_lengths": [len(g.moves) for g in against_bots],
                "bot_wins": sum(g.winner is not None and g.winner != g.bot_side for g in against_bots),
                "bot_draws": sum(g.winner is None for g in against_bots),
                "early_finishes": sum(g.adjudicated for g in games), "sims": sims,
                "requests": evaluator.requests, "computed": evaluator.computed,
                "seconds": time.time() - started}

    agent = NetAgent(evaluator, cfg["eval_sims"], cfg["c_puct"], cfg["fpu"], cfg["early_finish"], cfg["vcf_budget"], cfg["discount"],
                     cfg.get("root_safety", 0))
    openings = cfg.get("eval_openings")
    if openings is None:
        openings = 0 if geo.cells <= 25 else 2
    if task["opponent"] == "past":
        other = NetAgent(CachedEvaluator(remote_net(conn, 1, geo)), cfg["eval_sims"], cfg["c_puct"], cfg["fpu"],
                         cfg["early_finish"], cfg["vcf_budget"], cfg["discount"], cfg.get("root_safety", 0))
        score, played = play_match(agent, other, task["games"], rng, geo, opening_plies=openings, on_progress=on_progress)
    elif task["opponent"] == "pure":
        # постоянный сильный ориентир: только поиск и правила, никакого обучения
        other = NetAgent(uniform_net(geo), cfg["pure_sims"], cfg["c_puct"], cfg["fpu"], True, cfg["vcf_budget"])
        score, played = play_match(agent, other, task["games"], rng, geo, opening_plies=openings, on_progress=on_progress)
    else:
        score, played = play_match(agent, BOTS[task["opponent"]], task["games"], rng, geo, on_progress=on_progress)
    return {"opponent": task["opponent"], "score": score,
            "games": [{"moves": g.moves, "winner": g.winner, "net_first": g.a_first} for g in played]}


def worker_loop(cfg, conn, parent):
    geo = geometry(cfg["n"], cfg["d"], cfg.get("k"))
    while True:
        if not conn.poll(5):
            if os.getppid() != parent:
                return
            continue
        task = conn.recv()
        if task is None:
            return
        try:
            result = run_task(task, cfg, conn, geo)
        except Exception:
            result = {"error": traceback.format_exc()}
        conn.send_bytes(RESULT + pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
