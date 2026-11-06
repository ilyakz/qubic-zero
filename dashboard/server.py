#!/usr/bin/env python
"""Дашборд обучения: python dashboard/server.py, затем http://127.0.0.1:8765"""
import argparse
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
RESULTS = os.path.join(HERE, "..", "results", "findings.json")
MATCHES = os.path.join(HERE, "..", "results", "matches.json")
MODELS = os.path.join(HERE, "..", "docs", "models", "index.json")
_gpu_cache = {"at": 0.0, "data": None}
_system_cache = {"at": 0.0, "data": None, "times": None}
_summaries = {}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_metrics(path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass  # строка дописывается прямо сейчас
    except OSError:
        pass
    return records


def process_alive(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"train.py" in f.read()
    except (OSError, TypeError):
        return False


def gpu_state():
    if time.time() - _gpu_cache["at"] > 3:
        _gpu_cache["at"] = time.time()
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout
            temp, util, used, total = [float(v) for v in out.split(",")]
            _gpu_cache["data"] = {"temp": temp, "util": util, "used": used, "total": total}
        except Exception:
            _gpu_cache["data"] = None
    return _gpu_cache["data"]


def cpu_times():
    """Для каждого процессора из /proc/stat: всего тиков и сколько из них простой."""
    times = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            parts = line.split()
            values = [int(v) for v in parts[1:9]]
            times[parts[0]] = (sum(values), values[3] + values[4])
    return times


def busy_share(now, before):
    total = now[0] - before[0]
    return round(100 * (1 - (now[1] - before[1]) / total), 1) if total > 0 else 0.0


def physical_cores():
    pairs, physical = set(), None
    with open("/proc/cpuinfo") as f:
        for line in f:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                physical = value.strip()
            elif key == "core id":
                pairs.add((physical, value.strip()))
    return len(pairs) or None


def system_state():
    """Загрузка процессора и памяти по данным /proc: на других системах показать нечего."""
    if time.time() - _system_cache["at"] < 2:
        return _system_cache["data"]
    _system_cache["at"] = time.time()
    try:
        times = cpu_times()
        before = _system_cache["times"] or {name: (0, 0) for name in times}
        _system_cache["times"] = times
        memory = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                memory[key] = int(value.split()[0]) / 1024 ** 2  # ГБ
        names = sorted((n for n in times if n != "cpu"), key=lambda n: int(n[3:]))
        _system_cache["data"] = {
            "cpu": busy_share(times["cpu"], before["cpu"]), "per_core": [busy_share(times[n], before[n]) for n in names],
            "load": round(os.getloadavg()[0], 1), "threads": len(names), "cores": physical_cores(),
            "ram_total": round(memory["MemTotal"], 1), "ram_used": round(memory["MemTotal"] - memory["MemAvailable"], 1),
            "swap_total": round(memory.get("SwapTotal", 0), 1), "swap_used": round(memory.get("SwapTotal", 0) - memory.get("SwapFree", 0), 1)}
    except (OSError, ValueError, KeyError, IndexError):
        _system_cache["data"] = None
    return _system_cache["data"]


def run_names():
    try:
        return sorted(name for name in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, name, "status.json")))
    except OSError:
        return []


def run_status(name):
    folder = os.path.join(RUNS, name)
    status = read_json(os.path.join(folder, "status.json"))
    if status and status.get("state") == "running" and not process_alive(status.get("pid")):
        status["state"] = "crashed"
    return status


def run_state(name):
    folder = os.path.join(RUNS, name)
    status = run_status(name)
    if status is None:
        return None
    metrics = read_metrics(os.path.join(folder, "metrics.jsonl"))
    first_move = next((m["first_move"] for m in reversed(metrics) if m.get("first_move")), None)
    for record in metrics:
        record.pop("first_move", None)
    return {"name": name, "status": status, "config": read_json(os.path.join(folder, "config.json")), "metrics": metrics,
            "last_game": read_json(os.path.join(folder, "last_game.json")), "first_move": first_move,
            "perfect": read_json(os.path.join(folder, "perfect_check.json")),
            "stop_requested": os.path.exists(os.path.join(folder, "STOP"))}


def percent(wins, draws, losses):
    games = wins + draws + losses
    return round(100 * (wins + draws / 2) / games, 1) if games else None


def summarise_metrics(path):
    points, seconds = [], []
    for record in read_metrics(path):
        if record.get("iter_seconds"):
            seconds.append(record["iter_seconds"])
        results = record.get("eval")
        if not results:
            continue
        point = {"iter": record["iter"]}
        for opponent in ("tsp", "pure", "hsp", "osp"):
            score = results.get(opponent)
            if score:
                point[opponent] = percent(score["win"], score["draw"], score["loss"])
                point[opponent + "_first"] = percent(*score["as_first"])
                point[opponent + "_second"] = percent(*score["as_second"])
        points.append(point)
    recent = seconds[-8:]
    return {"points": points, "iter_seconds": sum(recent) / len(recent) if recent else None}


def run_summary(name):
    """Сводка одного запуска для общей таблицы. Метрики читаются заново, только когда файл изменился."""
    folder = os.path.join(RUNS, name)
    status = run_status(name)
    if status is None:
        return None
    path = os.path.join(folder, "metrics.jsonl")
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0
    cached = _summaries.get(name)
    if not cached or cached["stamp"] != stamp:
        cached = {"stamp": stamp, **summarise_metrics(path)}
        _summaries[name] = cached
    perfect = read_json(os.path.join(folder, "perfect_check.json")) or {}
    return {"name": name, "status": status, "config": read_json(os.path.join(folder, "config.json")) or {},
            "points": cached["points"], "iter_seconds": cached["iter_seconds"],
            "updated": os.path.getmtime(os.path.join(folder, "status.json")),
            "first_move_value": perfect.get("first_move_value"), "every_position": perfect.get("every_position"),
            "stop_requested": os.path.exists(os.path.join(folder, "STOP"))}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            names = run_names()
            wanted = parse_qs(url.query).get("run", [None])[0]
            statuses = {name: run_status(name) for name in names}
            self._json({"now": time.time(), "gpu": gpu_state(), "statuses": statuses,
                        "run": run_state(wanted) if wanted in names else None})
        elif url.path == "/api/overview":
            self._json({"now": time.time(), "gpu": gpu_state(), "system": system_state(), "runs": [s for s in map(run_summary, run_names()) if s],
                        "models": read_json(MODELS) or {}, "matches": read_json(MATCHES) or []})
        elif url.path == "/api/experiments":
            self._json({"findings": read_json(RESULTS), "runs": {name: run_state(name) for name in run_names()}})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        origin = self.headers.get("Origin")
        if self.path != "/api/stop" or (origin and self.headers.get("Host") not in origin):
            self._send(403, b"forbidden", "text/plain")
            return
        # останавливаем только выбранный запуск: рядом могут идти другие
        length = int(self.headers.get("Content-Length") or 0)
        try:
            name = json.loads(self.rfile.read(length) or b"{}").get("run")
        except ValueError:
            name = None
        state = read_json(os.path.join(RUNS, name, "status.json")) if name in run_names() else None
        if not state or state.get("state") != "running":
            self._send(400, b"no such running run", "text/plain")
            return
        open(os.path.join(RUNS, name, "STOP"), "w").close()
        self._send(200, b"{}", "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Дашборд: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
