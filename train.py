#!/usr/bin/env python
"""Обучение AlphaZero для кубических крестиков-ноликов 4×4×4.

Цикл: самоигра -> обучение сети на партиях -> проверка силы против ботов.
Прогресс пишется в runs/<имя>/, его показывает dashboard/server.py.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
from collections import deque
from multiprocessing.connection import wait

import numpy as np

from qubic.game import geometry
from qubic.worker import PROGRESS, REQUEST, worker_loop

# Игровые процессы импортируют этот файл заново, а torch им не нужен и только занял бы память.
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    from qubic.net import Evaluator, build_net, occupied_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
OPPONENTS = ("osp", "hsp", "tsp", "pure")
SNAPSHOT_EVERY = 5
BATCH_WAIT = 0.004  # секунд ждать запросы остальных процессов, прежде чем считать пачку


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="main", help="имя запуска, папка runs/<имя>")
    p.add_argument("--resume", action="store_true", help="продолжить запуск с последней итерации")
    p.add_argument("--legacy-labels", action="store_true",
                   help="контрольный запуск: метки оценки с ошибкой 2019 года")
    p.add_argument("--no-early-finish", dest="early_finish", action="store_false",
                   help="доигрывать партии до конца, даже когда форсированная победа уже есть")
    p.add_argument("--n", type=int, default=4, help="сторона доски")
    p.add_argument("--k", type=int, default=None, help="сколько нужно собрать в ряд; по умолчанию во всю сторону доски")
    p.add_argument("--d", type=int, default=3, help="число измерений")
    p.add_argument("--init-replay", default=None, help="начать с партий другого запуска, файл replay.npz: иначе первые итерации переобучаются на горстке позиций")
    p.add_argument("--init-from", default=None, help="начать с весов другого запуска: сеть linenet подходит к любому размеру")
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--workers", type=int, default=8, help="процессов, играющих партии")
    p.add_argument("--arch", choices=("resnet", "lines", "net2019", "linenet", "linenet2", "conv2d", "conv4d", "conv4d_lines", "hybrid"), default="resnet",
                   help="resnet — свёртки, lines — свёртки с признаками линий, net2019 — сеть версии 2019 года, linenet — сеть на линиях, linenet2 — она же с состоянием линий")
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--games", type=int, default=256, help="партий самоигры за итерацию")
    p.add_argument("--parallel", type=int, default=64, help="одновременных партий на процесс")
    p.add_argument("--sims", type=int, default=200, help="симуляций поиска на ход в самоигре")
    p.add_argument("--temperature-plies", type=int, default=6)
    p.add_argument("--discount", type=float, default=1.0, help="ценность победы падает на каждый ход до конца партии, 1 — не падает")
    p.add_argument("--root-safety", type=int, default=0, help="убирать в корне ходы, отдающие серию шахов; бюджет проверки, 0 — выключено")
    p.add_argument("--random-plies", type=int, default=0, help="до скольких случайных ходов в начале партии самоигры, не больше 6")
    p.add_argument("--vcf-budget", type=int, default=400, help="узлов поиска серии шахов на позицию в дереве, 0 — не искать")
    p.add_argument("--gumbel-width", type=int, default=0, help="сколько ходов рассматривать в корне по Gumbel AlphaZero; 0 — обычный поиск")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--noise-alpha", type=float, default=0.3)
    p.add_argument("--noise-eps", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=200, help="шагов обучения за итерацию")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--league", default="", help='доля партий против ботов, например "tsp=0.5,hsp=0.1"')
    p.add_argument("--eval-against", default=",".join(OPPONENTS), help="с кем проверять силу")
    p.add_argument("--league-bot-first", type=float, default=0.5, help="в какой доле партий с ботом он ходит первым")
    p.add_argument("--rival-sims", type=int, default=25, help="симуляций у постоянного соперника в партиях лиги, он там называется rival")
    p.add_argument("--rival", default=None, help="постоянный соперник на каждой проверке силы: runs/имя@итерация; в счёте он называется past")
    p.add_argument("--patience", type=int, default=0, help="остановиться, если столько проверок подряд нет роста силы; 0 — не останавливаться")
    p.add_argument("--min-delta", type=float, default=1.5, help="какой прирост в процентных пунктах считать ростом")
    p.add_argument("--lr-drops", type=int, default=0,
                   help="сколько раз на плато снизить шаг обучения, прежде чем остановиться; 0 — шаг идёт по расписанию")
    p.add_argument("--strength-against", default="tsp,pure", help="по каким соперникам считать силу для остановки")
    p.add_argument("--lr", type=float, default=None, help="постоянная скорость обучения вместо расписания")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=300000, help="максимум позиций в буфере")
    p.add_argument("--eval-every", type=int, default=3, help="раз во сколько итераций проверять силу")
    p.add_argument("--eval-games", type=int, default=40, help="партий против каждого бота")
    p.add_argument("--eval-sims", type=int, default=100, help="симуляций на ход при проверке силы")
    p.add_argument("--eval-openings", type=int, default=None,
                   help="случайных ходов в начале партии против поиска; по умолчанию 2, а до 25 клеток 0: там такое начало бывает уже проигранным")
    p.add_argument("--pure-sims", type=int, default=1000, help="симуляций на ход у соперника без обученной сети")
    p.add_argument("--past-every", type=int, default=15, help="раз во сколько итераций играть со своей старой версией")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


BASE_LR, LR_DROP = 1e-3, 0.3


def learning_rate(iteration):
    return 1e-3 if iteration <= 60 else 3e-4 if iteration <= 150 else 1e-4


def replay_limit(iteration, window):
    return min(window, 40000 + iteration * 3000)


def split_games(total, parts):
    """Делит партии между процессами чётными порциями: сеть играет обеими сторонами поровну."""
    base = (total // parts) // 2 * 2
    sizes = [base] * parts
    sizes[-1] += total - base * parts
    return [s for s in sizes if s > 0]


def read_metrics(path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return records


def load_rival(spec, device):
    run, _, iteration = spec.partition("@")
    with open(os.path.join(run, "config.json")) as f:
        rival = json.load(f)
    net = build_net(rival["arch"], geometry(rival.get("n", 4), rival.get("d", 3), rival.get("k")), rival["channels"], rival["blocks"]).to(device).eval()
    net.load_state_dict(torch.load(os.path.join(run, "nets", f"net_{int(iteration):04d}.pt"), map_location=device, weights_only=True))
    return net


class PlateauWatch:
    """Следит за силой сети и говорит, когда обучение перестало давать прирост.

    Одна проверка шумная, поэтому сравнивается среднее по нескольким последним. Лучшая версия —
    та, на которой это среднее было наибольшим; стоп, если patience проверок подряд нет роста.
    Застой может значить, что шаг обучения слишком велик, поэтому сначала шаг снижается drops раз.
    """

    def __init__(self, patience, min_delta=1.5, window=3, drops=0):
        self.patience, self.min_delta, self.window, self.drops = patience, min_delta, window, drops
        self.drops_done = 0
        self.scores = []
        self.best = None
        self.best_iter = None
        self.wait = 0

    def add(self, iteration, score):
        """Возвращает (новая ли это лучшая версия, пора ли останавливаться)."""
        self.scores.append(score)
        if len(self.scores) < self.window:
            return False, False
        mean = sum(self.scores[-self.window:]) / self.window
        if self.best is None or mean > self.best + self.min_delta:
            self.best, self.best_iter, self.wait = mean, iteration, 0
            return True, False
        self.wait += 1
        if not self.patience or self.wait < self.patience:
            return False, False
        if self.drops_done < self.drops:
            self.drops_done += 1
            self.wait = 0
            return False, False
        return False, True


class Pool:
    """Игровые процессы и их запросы к сети: всё, что накопилось, считается одной пачкой."""

    def __init__(self, cfg, status):
        ctx = mp.get_context("spawn")
        self.status = status
        self.words = geometry(cfg["n"], cfg["d"], cfg.get("k")).words
        self.conns, self.workers = [], []
        self.batches = self.positions = 0
        for _ in range(cfg["workers"]):
            here, there = ctx.Pipe()
            worker = ctx.Process(target=worker_loop, daemon=True, args=(cfg, there, os.getpid()))
            worker.start()
            there.close()
            self.conns.append(here)
            self.workers.append(worker)

    def _serve(self, evaluator, group):
        arrays = [np.frombuffer(payload, "<u8").reshape(-1, self.words) for _, payload in group]
        owns = np.concatenate([a[:len(a) // 2] for a in arrays])
        opps = np.concatenate([a[len(a) // 2:] for a in arrays])
        probs, values = evaluator(owns, opps)
        self.batches += 1
        self.positions += len(owns)
        start = 0
        for (conn, _), a in zip(group, arrays):
            end = start + len(a) // 2
            conn.send_bytes(probs[start:end].tobytes() + values[start:end].tobytes())
            start = end

    def run(self, tasks, evaluators):
        """Выполняет задачи и отвечает на запросы к сети. evaluators: {номер сети: Evaluator}."""
        waiting = deque(tasks)
        busy = set()
        for conn in self.conns:
            if waiting:
                conn.send(waiting.popleft())
                busy.add(conn)
        self.batches = self.positions = 0
        done = 0
        collected = []
        requests = {}  # процесс -> его запрос, ждущий ответа сети
        oldest = None
        while len(collected) < len(tasks):
            listening = [conn for conn in busy if conn not in requests]
            if listening:
                # Вызов карты стоит почти одинаково и для 10, и для 60 позиций, поэтому
                # недолго ждём, пока остальные процессы дошлют свои: пачка выходит крупнее.
                timeout = 1.0 if oldest is None else max(0.0, oldest + BATCH_WAIT - time.perf_counter())
                ready = wait(listening, timeout=timeout)
            else:
                ready = []
            if not ready and not requests:
                if not all(w.is_alive() for w in self.workers):
                    raise RuntimeError("один из игровых процессов завершился аварийно")
                self.status.set(done=done)
                continue
            for conn in ready:
                try:
                    message = conn.recv_bytes()
                except EOFError:
                    raise RuntimeError("один из игровых процессов завершился аварийно") from None
                kind = message[:1]
                if kind == REQUEST:
                    requests[conn] = message
                    if oldest is None:
                        oldest = time.perf_counter()
                elif kind == PROGRESS:
                    done += int.from_bytes(message[1:], "little")
                else:
                    result = pickle.loads(message[1:])
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    collected.append(result)
                    if waiting:
                        conn.send(waiting.popleft())
                    else:
                        busy.discard(conn)
            everyone_asked = requests and all(conn in requests for conn in busy)
            if requests and (everyone_asked or time.perf_counter() - oldest >= BATCH_WAIT):
                groups = {}
                for conn, message in requests.items():
                    groups.setdefault(message[1], []).append((conn, message[2:]))
                for net_id, group in groups.items():
                    self._serve(evaluators[net_id], group)
                requests = {}
                oldest = None
            self.status.set(done=done)
        return collected

    def close(self):
        for conn in self.conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()


# --- служебное ------------------------------------------------------------------------------

def atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Status:
    """Текущее состояние запуска для дашборда; пишется не чаще раза в секунду."""

    def __init__(self, path, **base):
        self.path = path
        self.data = dict(base)
        self.written = 0.0

    def set(self, force=False, **fields):
        phase_changed = "phase" in fields and fields["phase"] != self.data.get("phase")
        self.data.update(fields)
        if phase_changed:
            self.data["phase_started"] = time.time()
        if force or phase_changed or time.time() - self.written > 1.0:
            self.data["updated"] = time.time()
            atomic_json(self.path, self.data)
            self.written = time.time()


def merge_scores(results):
    scores = {}
    for r in results:
        s = scores.setdefault(r["opponent"], {"games": 0, "win": 0, "draw": 0, "loss": 0,
                                              "as_first": [0, 0, 0], "as_second": [0, 0, 0]})
        for key in ("games", "win", "draw", "loss"):
            s[key] += r["score"][key]
        for key in ("as_first", "as_second"):
            s[key] = [a + b for a, b in zip(s[key], r["score"][key])]
    return scores


def pick_game(results, iteration, geo):
    """Партия для показа: последняя решённая против самого сильного бота."""
    for name in ("pure", "tsp", "hsp", "osp"):
        games = [g for r in results if r["opponent"] == name for g in r["games"]]
        if games:
            game = next((g for g in reversed(games) if g["winner"] is not None), games[-1])
            return {"iter": iteration, "opponent": name, "net_first": game["net_first"], "moves": game["moves"],
                    "winner": game["winner"], "line": geo.winning_line(game["moves"]) if game["winner"] is not None else []}
    return None


def value_check(net, device, arrays, geo):
    """Понимает ли сеть, кто выигрывает: прогноз v на свежих партиях, которых она ещё не видела."""
    preds = []
    with torch.inference_mode():
        for i in range(0, len(arrays["own"]), 256):
            x = torch.from_numpy(geo.encode_packed(arrays["own"][i:i + 256], arrays["opp"][i:i + 256])).to(device)
            preds.append(net(x)[1].float().cpu().numpy())
    preds = np.concatenate(preds)
    truth = arrays["true_z"]
    decisive = truth != 0
    accuracy = float((np.sign(preds[decisive]) == np.sign(truth[decisive])).mean()) if decisive.any() else None
    corr = float(np.corrcoef(preds, truth)[0, 1]) if truth.std() > 0 and preds.std() > 0 else 0.0
    return {"accuracy": accuracy, "corr": corr, "positions": int(len(truth))}


def sample_batch(buffer, batch, rng, device, geo):
    idx = rng.integers(len(buffer["own"]), size=batch)
    perm = geo.symmetries[rng.integers(len(geo.symmetries), size=batch)]
    own = np.take_along_axis(geo.unpack(buffer["own"][idx]), perm, axis=1)
    opp = np.take_along_axis(geo.unpack(buffer["opp"][idx]), perm, axis=1)
    pi = np.take_along_axis(buffer["pi"][idx], perm, axis=1)
    x = np.stack((own, opp), axis=1).reshape((-1, 2) + geo.shape)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(pi).to(device, non_blocking=True),
            torch.from_numpy(buffer["z"][idx]).to(device, non_blocking=True))


def train_network(net, optimizer, scaler, buffer, cfg, iteration, rng, device, status, geo):
    if cfg.get("lr_drops"):
        lr = (cfg["lr"] or BASE_LR) * LR_DROP ** cfg.get("lr_drops_done", 0)
    else:
        lr = cfg["lr"] or learning_rate(iteration)
    for group in optimizer.param_groups:
        group["lr"] = lr
    net.train()
    totals = np.zeros(3)
    for step in range(cfg["steps"]):
        x, pi, z = sample_batch(buffer, cfg["batch"], rng, device, geo)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits, v = net(x)
        logp = F.log_softmax(logits.float().masked_fill(occupied_mask(x), -1e4), dim=1)
        policy_loss = -(pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(v.float(), z)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        entropy = -(pi * torch.log(pi.clamp_min(1e-8))).sum(dim=1).mean()
        totals += (policy_loss.item(), value_loss.item(), (policy_loss - entropy).item())
        status.set(done=step + 1)
    net.eval()
    policy, value, kl = totals / cfg["steps"]
    return {"policy_loss": float(policy), "value_loss": float(value), "kl": float(kl),
            "lr": lr, "steps": cfg["steps"], "buffer": int(len(buffer["own"]))}


def eval_tasks(cfg, seeds, with_past=False):
    tasks = []
    for name in cfg["eval_against"].split(",") + (["past"] if with_past else []):
        for games in split_games(cfg["eval_games"], cfg["workers"]):
            tasks.append({"kind": "match", "opponent": name, "games": games, "seed": seeds()})
    return tasks


def first_move_view(net, device, geo):
    priors, values = Evaluator(net, device, geo)([0], [0])
    return {"policy": [round(float(p), 4) for p in priors[0]], "value": round(float(values[0]), 3)}


def main():
    args = parse_args()
    cfg = vars(args).copy()
    cfg["league_shares"] = {name: float(share) for name, share in (item.split("=") for item in args.league.split(",") if item)}
    run_dir = os.path.join(ROOT, "runs", args.run)
    nets_dir = os.path.join(run_dir, "nets")
    os.makedirs(nets_dir, exist_ok=True)
    paths = {name: os.path.join(run_dir, name) for name in
             ("config.json", "status.json", "metrics.jsonl", "last_game.json", "checkpoint.pt", "replay.npz", "STOP")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    geo = geometry(args.n, args.d, args.k)
    net = build_net(args.arch, geo, args.channels, args.blocks).to(device).eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate(1), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start_iter = 1
    buffer = None
    elapsed_before = 0.0
    previous = read_json(paths["config.json"]) or {}
    speedup_from = previous.get("speedup_from_iter")
    if args.resume and os.path.exists(paths["checkpoint.pt"]):
        state = torch.load(paths["checkpoint.pt"], map_location=device, weights_only=True)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_iter = state["iter"] + 1
        if os.path.exists(paths["replay.npz"]):
            with np.load(paths["replay.npz"]) as data:
                buffer = {key: data[key] for key in data.files}
            for key in ("own", "opp"):
                if buffer[key].ndim == 1:  # запуск начат, когда доска умещалась в одно число
                    buffer[key] = buffer[key][:, None]
        if os.path.exists(paths["metrics.jsonl"]):
            with open(paths["metrics.jsonl"]) as f:
                elapsed_before = json.loads(f.readlines()[-1])["elapsed"]
        if previous and "early_finish" not in previous and args.early_finish:
            speedup_from = start_iter  # запуск начинался на старом коде, дашборд отметит границу
    elif os.path.exists(paths["metrics.jsonl"]):
        sys.exit(f"Запуск {args.run} уже существует: продолжить — --resume, начать заново — другое --run")
    elif args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=True)
        net.load_state_dict(state["net"] if "net" in state else state)
        if args.init_replay:
            with np.load(args.init_replay) as data:
                buffer = {key: data[key] for key in data.files}

    atomic_json(paths["config.json"], {**cfg, "params": sum(p.numel() for p in net.parameters()),
                                       "speedup_from_iter": speedup_from})
    if os.path.exists(paths["STOP"]):
        os.remove(paths["STOP"])
    run_started = time.time() - elapsed_before
    status = Status(paths["status.json"], state="running", pid=os.getpid(), run=args.run, started=run_started,
                    max_iters=args.max_iters, iter=start_iter - 1, phase="starting", done=0, total=0)
    status.set(force=True)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, start_iter]))
    seeds = lambda: int(rng.integers(2 ** 62))
    evaluators = {0: Evaluator(net, device, geo)}
    past_net = None
    if args.rival:
        evaluators[1] = Evaluator(load_rival(args.rival, device), device, geo)

    def log(record):
        with open(paths["metrics.jsonl"], "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_replay():
        tmp = paths["replay.npz"] + ".tmp.npz"
        np.savez(tmp, **buffer)
        os.replace(tmp, paths["replay.npz"])

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pool = Pool(cfg, status)
    final_state = "crashed"
    watch = PlateauWatch(args.patience, args.min_delta, drops=args.lr_drops)
    if start_iter > 1:  # продолжение: лучшая версия и счётчик застоя не должны забыться
        for record in read_metrics(paths["metrics.jsonl"]):
            if record.get("strength") is not None and record["iter"] < start_iter:
                watch.add(record["iter"], record["strength"])
    try:
        if start_iter == 1:
            status.set(phase="baseline", iter=0, done=0, total=args.eval_games * len(args.eval_against.split(",")), force=True)
            results = pool.run(eval_tasks(cfg, seeds), evaluators)
            log({"iter": 0, "time": time.time(), "elapsed": time.time() - run_started,
                 "eval": merge_scores(results), "first_move": first_move_view(net, device, geo)})
            game = pick_game(results, 0, geo)
            if game:
                atomic_json(paths["last_game.json"], game)

        for iteration in range(start_iter, args.max_iters + 1):
            iter_started = time.time()

            cfg["lr_drops_done"] = watch.drops_done

            # 1. Самоигра
            status.set(phase="selfplay", iter=iteration, done=0, total=args.games, iter_started=iter_started, force=True)
            results = pool.run([{"kind": "selfplay", "games": games, "seed": seeds()}
                                for games in split_games(args.games, args.workers)], evaluators)
            selfplay_seconds = time.time() - iter_started
            arrays = [r["arrays"] for r in results if r["arrays"] is not None]
            new = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
            winners = [w for r in results for w in r["winners"]]
            lengths = [n for r in results for n in r["lengths"]] or [0]
            bot_games = sum(r["bot_games"] for r in results)
            bot_score = sum(r["bot_wins"] + r["bot_draws"] / 2 for r in results)
            bot_lengths = [n for r in results for n in r["bot_lengths"]]
            sims = sum(r["sims"] for r in results)
            requests = sum(r["requests"] for r in results)
            computed = sum(r["computed"] for r in results)
            early_finishes = sum(r["early_finishes"] for r in results)
            avg_batch = pool.positions / max(pool.batches, 1)

            # 2. Проверка оценки на партиях, которых сеть ещё не видела, затем обучение
            check = value_check(net, device, new, geo)
            limit = replay_limit(iteration, args.window)
            fresh = {key: new[key] for key in ("own", "opp", "pi", "z")}
            if buffer is None:
                buffer = fresh
            else:
                buffer = {key: np.concatenate((buffer[key], fresh[key]))[-limit:] for key in fresh}
            train_started = time.time()
            status.set(phase="training", done=0, total=args.steps, force=True)
            train_stats = train_network(net, optimizer, scaler, buffer, cfg, iteration, rng, device, status, geo)
            train_stats["seconds"] = time.time() - train_started

            # 3. Проверка силы, не на каждой итерации
            past = None
            if args.past_every and iteration % args.past_every == 0:
                candidate = os.path.join(nets_dir, f"net_{iteration - args.past_every:04d}.pt")
                past = candidate if os.path.exists(candidate) else None
            record_eval = {}
            game = None
            if iteration % args.eval_every == 0 or iteration == args.max_iters or past:
                eval_started = time.time()
                if args.rival:
                    past = args.rival
                elif past:
                    if past_net is None:
                        past_net = build_net(args.arch, geo, args.channels, args.blocks).to(device).eval()
                        evaluators[1] = Evaluator(past_net, device, geo)
                    past_net.load_state_dict(torch.load(past, map_location=device, weights_only=True))
                status.set(phase="evaluation", done=0, force=True,
                           total=args.eval_games * (len(args.eval_against.split(",")) + (1 if past else 0)))
                results = pool.run(eval_tasks(cfg, seeds, with_past=bool(past)), evaluators)
                game = pick_game(results, iteration, geo)
                record_eval = {"eval": merge_scores(results), "eval_seconds": time.time() - eval_started,
                               "past_gap": args.past_every if past else None}

            stop_on_plateau = False
            if record_eval:
                names = [name for name in args.strength_against.split(",") if name in record_eval["eval"]]
                score = sum(100 * (record_eval["eval"][name]["win"] + record_eval["eval"][name]["draw"] / 2) / record_eval["eval"][name]["games"]
                            for name in names) / max(len(names), 1)
                is_best, stop_on_plateau = watch.add(iteration, score)
                if is_best:
                    torch.save(net.state_dict(), os.path.join(nets_dir, "best.pt"))
                record_eval.update(strength=score, best_iter=watch.best_iter, plateau_wait=watch.wait)
                status.set(best_iter=watch.best_iter, plateau_wait=watch.wait, patience=args.patience,
                           lr_drops=args.lr_drops, lr_drops_done=watch.drops_done, force=True)

            if iteration % SNAPSHOT_EVERY == 0:
                torch.save(net.state_dict(), os.path.join(nets_dir, f"net_{iteration:04d}.pt"))
            tmp = paths["checkpoint.pt"] + ".tmp"
            torch.save({"iter": iteration, "net": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict()}, tmp)
            os.replace(tmp, paths["checkpoint.pt"])
            stopping = os.path.exists(paths["STOP"]) or stop_on_plateau
            if iteration % SNAPSHOT_EVERY == 0 or iteration == args.max_iters or stopping:
                save_replay()

            log({"iter": iteration, "time": time.time(), "elapsed": time.time() - run_started,
                 "iter_seconds": time.time() - iter_started,
                 "selfplay": {"games": len(winners), "x_wins": winners.count(0), "o_wins": winners.count(1),
                              "draws": winners.count(None), "avg_length": float(np.mean(lengths)),
                              "positions": check["positions"], "sims": sims,
                              "early_finishes": early_finishes,
                              "sims_per_s": sims / max(selfplay_seconds, 1e-3), "seconds": selfplay_seconds,
                              "cache_hit": 1 - computed / max(requests, 1), "avg_batch": avg_batch,
                              "bot_games": bot_games, "bot_win_rate": 100 * bot_score / bot_games if bot_games else None,
                              "bot_avg_length": float(np.mean(bot_lengths)) if bot_lengths else None},
                 "value_check": check, "train": train_stats, **record_eval,
                 "first_move": first_move_view(net, device, geo)})
            if game:
                atomic_json(paths["last_game.json"], game)

            if stopping:
                if os.path.exists(paths["STOP"]):
                    os.remove(paths["STOP"])
                final_state = "plateau" if stop_on_plateau else "stopped"
                break
        else:
            final_state = "finished"
    except SystemExit:
        final_state = "stopped"
        raise
    except BaseException:
        status.set(error=traceback.format_exc(), force=True)
        raise
    finally:
        status.set(state=final_state, phase="idle", done=0, total=0, force=True)
        pool.close()


if __name__ == "__main__":
    main()
