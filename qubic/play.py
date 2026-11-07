"""Партии: самоигра для обучения и матчи для проверки силы, все партии идут параллельно."""
from dataclasses import dataclass, field

import numpy as np

from .game import outcome_labels
from .mcts import Node, gumbel_search, search, visit_policy


@dataclass
class Game:
    root: Node
    moves: list = field(default_factory=list)
    plies: list = field(default_factory=list)
    owns: list = field(default_factory=list)
    opps: list = field(default_factory=list)
    pis: list = field(default_factory=list)
    finished: bool = False
    winner: int | None = None
    adjudicated: bool = False
    opening: int = 0
    bot: object = None
    bot_side: int = 0

    def apply(self, cell, geo):
        """Делает ход; поддерево поиска переходит к следующей позиции."""
        ply = len(self.moves)
        me = self.root.own | 1 << cell
        self.moves.append(cell)
        if geo.completes_line(me, cell):
            self.finished, self.winner = True, ply % 2
        elif me | self.root.opp == geo.full:
            self.finished = True
        else:
            self.root = self.root.child_for_move(cell)


def _new_game():
    return Game(Node(0, 0))


def _pick_move(root, temperature, rng):
    visits = root.N
    if temperature > 0:
        index = int(rng.choice(len(visits), p=visits / visits.sum()))
    else:
        index = int((visits + rng.random(len(visits)) * 1e-3).argmax())
    return root.moves[index]


def _finish_if_decided(game, geo):
    """Заканчивает партию, если исход уже доказан: двойной угрозой или серией шахов.

    Доигрывать такие партии незачем: это половина всех считаемых ходов, а слабая сеть
    ещё и упускает выигрыш, из-за чего позиции получают неверный исход. В обучение идёт
    сама позиция с доказанной победой, целью политики становится первый ход серии.
    """
    root = game.root
    ply = len(game.moves)
    if root.terminal == -1.0:
        game.finished, game.winner, game.adjudicated = True, (ply + 1) % 2, True
        return True
    cell = root.forced if root.terminal == 1.0 else geo.rules.find_vcf(root.own, root.opp)
    if cell is None or geo.completes_line(root.own | 1 << cell, cell):
        return False
    pi = np.zeros(geo.cells, dtype=np.float32)
    pi[cell] = 1.0
    game.plies.append(ply)
    game.owns.append(root.own)
    game.opps.append(root.opp)
    game.pis.append(pi)
    game.finished, game.winner, game.adjudicated = True, ply % 2, True
    return True


def self_play(evaluator, total_games, parallel, sims, rng, geo, temperature_plies=12, c_puct=1.5,
              fpu=0.2, noise_alpha=0.3, noise_eps=0.25, on_progress=None, early_finish=True, vcf_budget=0,
              random_plies=0, discount=1.0, root_safety=0, league=(), bot_first=0.5, gumbel_width=0):
    """Играет total_games партий сети и возвращает их записи.

    random_plies: партия начинается со случайного числа (от 0 до random_plies) случайных ходов,
    в обучение они не идут. league: список пар (бот, доля партий). В таких партиях сеть играет
    с ботом случайной стороной и учится только на своих ходах: сама с собой она может не увидеть
    настоящей защиты, если первый игрок побеждает почти всегда. bot_first: доля таких партий,
    где бот ходит первым и сеть учится защищаться. gumbel_width > 0: поиск и цель политики по Gumbel AlphaZero.
    """
    active, finished = [], []
    started = 0
    simulations = 0
    while active or started < total_games:
        while len(active) < parallel and started < total_games:
            game = _new_game()
            game.opening = int(rng.integers(random_plies + 1))
            roll = rng.random()
            for bot, share in league:
                if roll < share:
                    game.bot, game.bot_side = bot, 0 if rng.random() < bot_first else 1
                    break
                roll -= share
            active.append(game)
            started += 1

        searching = []
        for game in active:
            while not game.finished:
                if len(game.moves) < game.opening:
                    free = geo.empty_cells(game.root.own, game.root.opp)
                    game.apply(free[int(rng.integers(len(free)))], geo)
                    continue
                if game.bot is not None and len(game.moves) % 2 == game.bot_side:
                    game.apply(int(game.bot(game.root.own, game.root.opp, rng, geo)), geo)
                    continue
                if not game.root.analysed:
                    game.root.analyse(geo)
                if early_finish and _finish_if_decided(game, geo):
                    break
                if game.root.forced is None:
                    searching.append(game)
                    break
                game.apply(game.root.forced, geo)

        if searching and gumbel_width:
            for game in searching:
                game.root = Node(game.root.own, game.root.opp)
                game.root.analyse(geo)
            scales = [1.0 if len(g.moves) < temperature_plies else 0.0 for g in searching]
            plans, done = gumbel_search(evaluator, [g.root for g in searching], sims, geo, gumbel_width, scales, rng,
                                        c_puct, fpu, vcf_budget, discount, root_safety)
            simulations += done
            for game, plan in zip(searching, plans):
                root = game.root
                game.plies.append(len(game.moves))
                game.owns.append(root.own)
                game.opps.append(root.opp)
                game.pis.append(plan.policy(geo))
                game.apply(plan.best_move(), geo)
        elif searching:
            simulations += search(evaluator, [g.root for g in searching], sims, geo, c_puct, fpu, rng,
                                  noise_alpha, noise_eps, vcf_budget, discount, root_safety)
            for game in searching:
                root = game.root
                ply = len(game.moves)
                game.plies.append(ply)
                game.owns.append(root.own)
                game.opps.append(root.opp)
                game.pis.append(visit_policy(root, geo))
                game.apply(_pick_move(root, 1.0 if ply < temperature_plies else 0.0, rng), geo)

        still_active = []
        for game in active:
            (finished if game.finished else still_active).append(game)
        if len(still_active) != len(active) and on_progress:
            on_progress(len(finished), total_games)
        active = still_active
    return finished, simulations


class NetAgent:
    """Сеть с поиском. use_vcf: доказанную серию шахов разыгрывает сразу, без поиска."""

    def __init__(self, evaluator, sims, c_puct=1.5, fpu=0.2, use_vcf=True, vcf_budget=0, discount=1.0, root_safety=0, gumbel_width=0):
        self.evaluator, self.sims, self.c_puct, self.fpu = evaluator, sims, c_puct, fpu
        self.use_vcf = use_vcf
        self.vcf_budget = vcf_budget
        self.discount = discount
        self.root_safety = root_safety
        self.gumbel_width = gumbel_width


def play_match(agent_a, agent_b, games, rng, geo, opening_plies=0, on_progress=None):
    """Матч: agent — NetAgent или функция-бот. Половину партий agent_a играет первым.

    Возвращает счёт для agent_a и записи партий с полем a_first.
    """
    active = []
    for i in range(games):
        game = _new_game()
        game.a_first = i % 2 == 0
        active.append(game)
    played = []

    def agent_for(game):
        a_turn = (len(game.moves) % 2 == 0) == game.a_first
        return agent_a if a_turn else agent_b

    while active:
        waiting = {}
        for game in active:
            while not game.finished:
                # новый корень на каждый ход: соперники не должны делить дерево
                game.root = Node(game.root.own, game.root.opp)
                agent = agent_for(game)
                if len(game.moves) < opening_plies:
                    game.root.analyse(geo)
                    free = geo.empty_cells(game.root.own, game.root.opp)
                    game.apply(game.root.forced if game.root.forced is not None
                               else free[int(rng.integers(len(free)))], geo)
                elif isinstance(agent, NetAgent):
                    game.root.analyse(geo)
                    cell = geo.rules.find_vcf(game.root.own, game.root.opp) if agent.use_vcf else None
                    if cell is None:
                        cell = game.root.forced
                    if cell is None:
                        waiting.setdefault(id(agent), (agent, []))[1].append(game)
                        break
                    game.apply(cell, geo)
                else:
                    game.apply(int(agent(game.root.own, game.root.opp, rng, geo)), geo)

        for agent, group in waiting.values():
            if agent.gumbel_width:
                plans, _ = gumbel_search(agent.evaluator, [g.root for g in group], agent.sims, geo, agent.gumbel_width, [0.0] * len(group), rng,
                                         agent.c_puct, agent.fpu, agent.vcf_budget, agent.discount, agent.root_safety)
                for game, plan in zip(group, plans):
                    game.apply(plan.best_move(), geo)
                continue
            search(agent.evaluator, [g.root for g in group], agent.sims, geo, agent.c_puct, agent.fpu,
                   vcf_budget=agent.vcf_budget, discount=agent.discount, root_safety=agent.root_safety)
            for game in group:
                game.apply(_pick_move(game.root, 0.0, rng), geo)

        still_active = []
        for game in active:
            if game.finished:
                played.append(game)
            else:
                still_active.append(game)
        if len(still_active) != len(active) and on_progress:
            on_progress(len(played), games)
        active = still_active

    return summarize(played), played


def summarize(played):
    score = {"games": len(played), "win": 0, "draw": 0, "loss": 0,
             "as_first": [0, 0, 0], "as_second": [0, 0, 0]}
    for game in played:
        side = "as_first" if game.a_first else "as_second"
        if game.winner is None:
            outcome = 1
            score["draw"] += 1
        elif (game.winner == 0) == game.a_first:
            outcome = 0
            score["win"] += 1
        else:
            outcome = 2
            score["loss"] += 1
        score[side][outcome] += 1
    return score


def replay_arrays(games, geo, legacy_labels=False, discount=1.0):
    """Записи партий в массивы для буфера обучения; метки считает outcome_labels."""
    owns, opps, pis, zs, true_zs = [], [], [], [], []
    for game in games:
        if not game.plies:
            continue
        owns.extend(game.owns)
        opps.extend(game.opps)
        pis.extend(game.pis)
        zs.append(outcome_labels(game.plies, game.winner, legacy_labels, discount, len(game.moves)))
        true_zs.append(outcome_labels(game.plies, game.winner))
    if not owns:
        return None
    return {
        "own": geo.pack(owns),
        "opp": geo.pack(opps),
        "pi": np.stack(pis),
        "z": np.concatenate(zs),
        "true_z": np.concatenate(true_zs),
    }
