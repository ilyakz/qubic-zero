"""Поиск по дереву (PUCT), в котором нейросеть считает листья всех партий одной пачкой."""
import math

import numpy as np

from . import fast

_ONE = np.ones(1)


class Node:
    """Вершина дерева. Значения N и W на рёбрах даны с точки зрения игрока, чей ход здесь."""
    __slots__ = ("own", "opp", "analysed", "terminal", "forced", "moves", "P", "N", "W",
                 "children", "visits", "q_sum")

    def __init__(self, own, opp):
        self.own = own
        self.opp = opp
        self.analysed = False
        self.terminal = None
        self.forced = None
        self.moves = None
        self.P = self.N = self.W = self.children = None
        self.visits = 0
        self.q_sum = 0.0

    def analyse(self, geo, vcf_budget=0):
        """Что следует прямо из правил, без нейросети.

        Есть свой выигрышный ход — позиция выиграна. Две и более угрозы соперника —
        проиграна, всё не закрыть. Одна угроза — ход вынужден, и сеть не нужна.
        vcf_budget > 0: ещё ищем выигрыш серией шахов, тогда позиция тоже выиграна.
        """
        self.analysed = True
        answer = geo.rules.analyse(self.own, self.opp, vcf_budget)
        code, cell = answer & 255, answer >> 8
        if code == fast.CODE_WIN or code == fast.CODE_VCF:
            self.terminal, self.forced = 1.0, cell
        elif code == fast.CODE_DRAW:
            self.terminal = 0.0
        elif code == fast.CODE_LOST:
            self.terminal, self.forced = -1.0, cell
        elif code == fast.CODE_FORCED:
            self.forced = cell
            self.moves = [cell]
            self.P = _ONE
            self.N = np.zeros(1)
            self.W = np.zeros(1)
            self.children = [None]

    def expand(self, priors, value, geo):
        self.moves = geo.empty_cells(self.own, self.opp)
        p = priors[self.moves].astype(np.float64)
        total = p.sum()
        self.P = p / total if total > 0 else np.full(len(p), 1.0 / len(p))
        self.N = np.zeros(len(p))
        self.W = np.zeros(len(p))
        self.children = [None] * len(p)
        self.q_sum = float(value)

    def child(self, index):
        node = self.children[index]
        if node is None:
            node = Node(self.opp, self.own | 1 << self.moves[index])
            self.children[index] = node
        return node

    def child_for_move(self, cell):
        """Следующая вершина после хода cell; поддерево переиспользуется, если уже есть."""
        if self.moves is not None and cell in self.moves:
            return self.child(self.moves.index(cell))
        return Node(self.opp, self.own | 1 << cell)


def _select(node, c_puct, fpu):
    if len(node.moves) == 1:
        return 0
    N = node.N
    unvisited_q = node.q_sum / (node.visits + 1) - fpu
    q = np.where(N > 0, node.W / (N + (N == 0)), unvisited_q)
    return int((q + c_puct * math.sqrt(node.visits + 1) * node.P / (1 + N)).argmax())


def _descend(root, geo, c_puct, fpu, vcf_budget):
    node = root
    path = []
    while True:
        if not node.analysed:
            node.analyse(geo, vcf_budget)
        if node.terminal is not None or node.moves is None:
            return node, path
        index = _select(node, c_puct, fpu)
        path.append((node, index))
        node = node.child(index)


def _backup(path, value, discount=1.0):
    for node, index in reversed(path):
        value = -value * discount
        node.N[index] += 1
        node.W[index] += value
        node.visits += 1
        node.q_sum += value


def _evaluate(evaluator, nodes, geo):
    priors, values = evaluator([n.own for n in nodes], [n.opp for n in nodes])
    for i, node in enumerate(nodes):
        node.expand(priors[i], values[i], geo)
    return values


def remove_unsafe_moves(root, geo, budget, top=64):
    """Убирает из корня ходы, после которых у соперника есть выигрыш серией шахов.

    Когда возможных ходов больше, чем симуляций, сам поиск проверить их все не успевает.
    Если опасны все проверенные ходы, ничего не меняем: позиция и так тяжёлая.
    """
    if root.moves is None or len(root.moves) < 2 or root.forced is not None:
        return
    order = np.argsort(-root.P)[:top]
    unsafe = {int(i) for i in order if geo.rules.find_vcf(root.opp, root.own | 1 << root.moves[i], budget) is not None}
    if not unsafe or len(unsafe) == len(order):
        return
    keep = [i for i in range(len(root.moves)) if i not in unsafe]
    root.visits -= int(root.N[list(unsafe)].sum())
    root.moves = [root.moves[i] for i in keep]
    root.children = [root.children[i] for i in keep]
    root.N, root.W = root.N[keep], root.W[keep]
    root.P = root.P[keep] / root.P[keep].sum()


def search(evaluator, roots, sims, geo, c_puct=1.5, fpu=0.2, rng=None, noise_alpha=0.3, noise_eps=0.0, vcf_budget=0,
           discount=1.0, root_safety=0):
    """Доводит число посещений каждого корня до sims. Возвращает число сделанных симуляций.

    evaluator(owns, opps) -> (priors[B][cells], values[B]). Корни с вынужденным ходом
    сюда передавать не нужно: искать там нечего.
    """
    for root in roots:
        if not root.analysed:
            root.analyse(geo, vcf_budget)
    unexpanded = [r for r in roots if r.moves is None and r.terminal is None]
    if unexpanded:
        _evaluate(evaluator, unexpanded, geo)
    if root_safety:
        for root in roots:
            remove_unsafe_moves(root, geo, root_safety)
    if noise_eps > 0:
        for root in roots:
            if root.moves is not None and len(root.moves) > 1:
                noise = rng.dirichlet([noise_alpha] * len(root.moves))
                root.P = (1 - noise_eps) * root.P + noise_eps * noise

    done = 0
    for _ in range(sims):
        leaves, paths = [], []
        for root in roots:
            if root.visits >= sims or root.terminal is not None:
                continue
            node, path = _descend(root, geo, c_puct, fpu, vcf_budget)
            done += 1
            if node.terminal is not None:
                _backup(path, node.terminal, discount)
            else:
                leaves.append(node)
                paths.append(path)
        if leaves:
            values = _evaluate(evaluator, leaves, geo)
            for path, value in zip(paths, values):
                _backup(path, float(value), discount)
    return done


def visit_policy(root, geo):
    """Доли посещений ходов корня как вектор длины cells — цель обучения политики."""
    pi = np.zeros(geo.cells, dtype=np.float32)
    pi[root.moves] = root.N / max(root.N.sum(), 1.0)
    return pi
