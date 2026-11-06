"""Правила k в ряд на доске n^d. Обычно k = n и линия идёт через всю доску: 3×3, 4×4×4, 4×4×4×4.
Если k меньше стороны, линий гораздо больше, и выходит 5×5 с четырьмя в ряд или гомоку: 15×15, пять в ряд.

Камни хранятся битовыми масками, целыми числами Python. Состояние дано с точки зрения
игрока, чей сейчас ход: own — его камни, opp — камни соперника.
Номер клетки — координаты в системе счисления по основанию n: для 4×4×4 это z*16 + y*4 + x.
Несколько камней подряд сверх k тоже считаются победой: любые k из них образуют линию.
"""
import functools
import itertools

import numpy as np

from .fast import Rules


class Geometry:
    def __init__(self, n, d, k=None):
        self.n, self.d, self.k = n, d, k or n
        self.name = "x".join([str(n)] * d) + (f"k{self.k}" if self.k != n else "")
        self.shape = (n,) * d
        self.cells = n ** d
        self.full = (1 << self.cells) - 1
        self.words = (self.cells + 63) // 64
        self.lines = self._build_lines()
        self.line_masks = [sum(1 << c for c in line) for line in self.lines]
        self.lines_through = [[m for m in self.line_masks if m >> c & 1] for c in range(self.cells)]
        self.degree = np.array([len(lines) for lines in self.lines_through])
        self.symmetries = self._build_symmetries()
        self.rules = Rules(self.line_masks, self.k, self.cells, self.words)

    def cell_index(self, coords):
        index = 0
        for coord in coords:
            index = index * self.n + coord
        return index

    def _build_lines(self):
        n, d, k = self.n, self.d, self.k
        lines = set()
        for start in itertools.product(range(n), repeat=d):
            for step in itertools.product((-1, 0, 1), repeat=d):
                if not any(step):
                    continue
                end = [start[i] + step[i] * (k - 1) for i in range(d)]
                if all(0 <= c < n for c in end):
                    lines.add(frozenset(self.cell_index([start[i] + step[i] * t for i in range(d)]) for t in range(k)))
        return sorted(sorted(line) for line in lines)

    def _build_symmetries(self):
        """Движения куба (d!·2^d штук) как перестановки клеток."""
        n, d = self.n, self.d
        perms = []
        coords = list(itertools.product(range(n), repeat=d))
        for axes in itertools.permutations(range(d)):
            for flips in itertools.product((False, True), repeat=d):
                perm = np.zeros(self.cells, dtype=np.int64)
                for point in coords:
                    moved = [point[axes[i]] for i in range(d)]
                    moved = [n - 1 - c if flips[i] else c for i, c in enumerate(moved)]
                    perm[self.cell_index(point)] = self.cell_index(moved)
                perms.append(perm)
        return np.stack(perms)

    def empty_cells(self, own, opp):
        free = self.full ^ (own | opp)
        if not free:
            return []
        cells = []
        for chunk, byte in enumerate(free.to_bytes(self.words * 8, "little")):
            if byte:
                cells += _BYTE_CELLS[chunk][byte]
        return cells

    def completes_line(self, stones, cell):
        """Замыкает ли камень в клетке cell какую-нибудь линию у игрока stones."""
        for mask in self.lines_through[cell]:
            if stones & mask == mask:
                return True
        return False

    def winning_cells(self, own, opp):
        """Клетки, ходом в которые own выигрывает немедленно."""
        cells = []
        for mask in self.line_masks:
            if mask & opp:
                continue
            rest = mask & ~own
            if rest and not rest & (rest - 1):
                cell = rest.bit_length() - 1
                if cell not in cells:
                    cells.append(cell)
        return cells

    def winning_line(self, moves):
        """Клетки линии, которой закончилась партия, либо пустой список."""
        if not moves:
            return []
        stones = sum(1 << c for c in moves[-1::-2])
        for line, mask in zip(self.lines, self.line_masks):
            if stones & mask == mask:
                return list(line)
        return []

    def pack(self, values):
        """Список масок в массив uint64 размера (B, words)."""
        if self.words == 1:
            return np.array(values, dtype=np.uint64).reshape(-1, 1)
        size = self.words * 8
        data = b"".join(v.to_bytes(size, "little") for v in values)
        return np.frombuffer(data, dtype="<u8").reshape(-1, self.words)

    def unpack(self, packed):
        """Массив (B, words) в биты (B, cells)."""
        as_bytes = packed.astype("<u8").view(np.uint8).reshape(len(packed), -1)
        return np.unpackbits(as_bytes, axis=1, bitorder="little", count=self.cells).astype(np.float32)

    def encode_packed(self, own, opp):
        """Вход сети: две плоскости, свои и чужие камни, размера (B, 2, n, ..., n)."""
        return np.stack((self.unpack(own), self.unpack(opp)), axis=1).reshape((-1, 2) + self.shape)

    def encode(self, owns, opps):
        return self.encode_packed(self.pack(owns), self.pack(opps))


_BYTE_CELLS = [[[8 * chunk + bit for bit in range(8) if byte >> bit & 1] for byte in range(256)] for chunk in range(64)]


@functools.lru_cache(maxsize=None)
def geometry(n=4, d=3, k=None):
    return Geometry(n, d, k)


def outcome_labels(plies, winner, legacy=False, discount=1.0, last_ply=0):
    """Цель обучения v для позиций партии с точки зрения игрока, чей ход в позиции.

    plies: номера полуходов (чётный — ходят крестики), winner: 0 — крестики,
    1 — нолики, None — ничья. Именно в этом месте была главная ошибка 2019 года.
    legacy=True воспроизводит её для контрольного запуска: метка зависит только
    от того, чей ход, и не зависит от исхода партии.
    """
    plies = np.asarray(plies)
    if legacy:
        return np.where(plies % 2 == 0, -1.0, 1.0).astype(np.float32)
    if winner is None:
        return np.zeros(len(plies), dtype=np.float32)
    labels = np.where(plies % 2 == winner, 1.0, -1.0)
    if discount < 1.0:
        # победа тем ценнее, чем ближе конец партии: проигрывающий тянет время, выигрывающий спешит
        labels = labels * discount ** np.maximum(last_ply - plies, 0)
    return labels.astype(np.float32)
