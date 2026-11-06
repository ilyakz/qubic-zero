#!/usr/bin/env python
"""Игра против сети на этом компьютере: python play.py, затем http://127.0.0.1:8766

Вся игра работает в браузере, этот скрипт только отдаёт страницу из папки docs.
"""
import argparse
import functools
import http.server
import os

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


class NoCache(http.server.SimpleHTTPRequestHandler):
    """Браузер не должен держать старую версию страницы после обновления кода."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    handler = functools.partial(NoCache, directory=DOCS)
    print(f"Игра: http://127.0.0.1:{args.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
