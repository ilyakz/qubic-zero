"""Запуск: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qubic import fast
from qubic.bots import BOTS, find_vcf_py, hsp_bot, osp_bot, tsp_bot
from qubic.game import geometry, outcome_labels
from qubic.mcts import Node, search, visit_policy
from qubic.play import NetAgent, play_match, replay_arrays, self_play
from qubic.worker import CachedEvaluator, uniform_net

QUBIC = geometry(4, 3)
ALL = [geometry(3, 3), QUBIC, geometry(5, 3), geometry(4, 4)]


def stones(*cells):
    return sum(1 << c for c in cells)


uniform = uniform_net(QUBIC)


def random_positions(geo, count, seed, longest=None):
    rng = np.random.default_rng(seed)
    found = []
    while len(found) < count:
        own, opp = 0, 0
        for _ in range(int(rng.integers(0, longest or geo.cells * 2 // 3))):
            free = geo.empty_cells(own, opp)
            cell = free[int(rng.integers(len(free)))]
            if geo.completes_line(own | 1 << cell, cell):
                break
            own, opp = opp, own | 1 << cell
        found.append((own, opp))
    return found


class Rules(unittest.TestCase):
    def test_line_counts_match_the_formula(self):
        for geo in ALL:
            self.assertEqual(len(geo.lines), ((geo.n + 2) ** geo.d - geo.n ** geo.d) // 2, geo.name)
            self.assertTrue(all(len(line) == geo.n for line in geo.lines))
            self.assertEqual(len(set(map(tuple, geo.lines))), len(geo.lines))

    def test_qubic_degrees(self):
        self.assertEqual(sorted(QUBIC.degree.tolist()), [4] * 48 + [7] * 16)

    def test_symmetries_keep_lines(self):
        for geo in ALL:
            expected = {frozenset(line) for line in geo.lines}
            self.assertEqual(len({p.tobytes() for p in geo.symmetries}), len(geo.symmetries))
            for perm in geo.symmetries[:: max(1, len(geo.symmetries) // 24)]:
                self.assertEqual({frozenset(int(perm[c]) for c in line) for line in geo.lines}, expected, geo.name)

    def test_symmetry_counts(self):
        self.assertEqual(len(QUBIC.symmetries), 48)
        self.assertEqual(len(geometry(4, 4).symmetries), 384)

    def test_winning_cells(self):
        own = stones(0, 1, 2)
        self.assertEqual(QUBIC.winning_cells(own, 0), [3])
        self.assertEqual(QUBIC.winning_cells(own, stones(3)), [])
        self.assertTrue(QUBIC.completes_line(own | 1 << 3, 3))
        self.assertFalse(QUBIC.completes_line(own, 2))

    def test_encode_planes(self):
        x = QUBIC.encode([1 << QUBIC.cell_index((1, 2, 3))], [1 << QUBIC.cell_index((3, 0, 1))])
        self.assertEqual(x.shape, (1, 2, 4, 4, 4))
        self.assertEqual(x[0, 0, 1, 2, 3], 1.0)
        self.assertEqual(x[0, 1, 3, 0, 1], 1.0)
        self.assertEqual(x.sum(), 2.0)

    def test_pack_round_trip_on_every_width(self):
        for geo in ALL:
            cells = [0, geo.cells // 2, geo.cells - 1]
            value = stones(*cells)
            bits = geo.unpack(geo.pack([value, 0]))
            self.assertEqual(bits.shape, (2, geo.cells))
            self.assertEqual(np.flatnonzero(bits[0]).tolist(), cells, geo.name)
            self.assertEqual(bits[1].sum(), 0)

    def test_empty_cells(self):
        for geo in ALL:
            for own, opp in random_positions(geo, 100, 1):
                self.assertEqual(geo.empty_cells(own, opp), [c for c in range(geo.cells) if not (own | opp) >> c & 1])
            self.assertEqual(geo.empty_cells(geo.full, 0), [])

    def test_winning_line(self):
        self.assertEqual(QUBIC.winning_line([0, 16, 1, 17, 2, 18, 3]), [0, 1, 2, 3])


class Labels(unittest.TestCase):
    """Ошибка 2019 года: метка не зависела от того, кто выиграл."""

    def test_winner_gets_plus_one(self):
        np.testing.assert_array_equal(outcome_labels([0, 1, 2, 3], 0), [1, -1, 1, -1])
        np.testing.assert_array_equal(outcome_labels([0, 1, 2, 3], 1), [-1, 1, -1, 1])

    def test_draw_is_zero(self):
        np.testing.assert_array_equal(outcome_labels([0, 1], None), [0, 0])

    def test_labels_depend_on_winner(self):
        self.assertFalse(np.array_equal(outcome_labels([0, 1, 2], 0), outcome_labels([0, 1, 2], 1)))

    def test_discount_makes_near_wins_worth_more(self):
        labels = outcome_labels([0, 1, 8, 9], 0, discount=0.9, last_ply=10)
        self.assertAlmostEqual(float(labels[2]), 0.81, places=5)
        self.assertAlmostEqual(float(labels[3]), -0.9, places=5)
        self.assertLess(abs(float(labels[0])), abs(float(labels[2])))

    def test_legacy_mode_reproduces_the_bug(self):
        for winner in (0, 1, None):
            np.testing.assert_array_equal(outcome_labels([0, 1, 2, 3], winner, legacy=True), [-1, 1, -1, 1])


class Search(unittest.TestCase):
    def test_win_available_is_terminal(self):
        node = Node(stones(0, 1, 2), stones(20, 21))
        node.analyse(QUBIC)
        self.assertEqual((node.terminal, node.forced), (1.0, 3))

    def test_single_threat_forces_the_block(self):
        node = Node(stones(20), stones(0, 1, 2))
        node.analyse(QUBIC)
        self.assertIsNone(node.terminal)
        self.assertEqual(node.forced, 3)

    def test_double_threat_is_lost(self):
        node = Node(stones(40), stones(0, 1, 2, 16, 32))
        node.analyse(QUBIC)
        self.assertEqual(node.terminal, -1.0)

    def test_search_prefers_move_that_wins_later(self):
        # ход 0 создаёт двойную угрозу: линии 0-1-2-3 и 0-16-32-48, значит соперник не отобьётся
        own, opp = stones(1, 2, 16, 32), stones(5, 10, 25, 42)
        root = Node(own, opp)
        search(uniform, [root], 400, QUBIC)
        self.assertEqual(int(np.argmax(visit_policy(root, QUBIC))), 0)

    def test_values_are_seen_from_the_side_to_move(self):
        # после хода 0 у соперника вынужденный проигрыш, значит оценка хода для нас положительна
        root = Node(stones(1, 2, 16, 32), stones(5, 10, 25, 42))
        search(uniform, [root], 400, QUBIC)
        self.assertGreater(root.W[root.moves.index(0)] / root.N[root.moves.index(0)], 0.9)

    def test_tree_sees_forced_win_of_the_opponent(self):
        # соперник серией шахов выигрывает везде, кроме ходов, которые ломают его серию
        own, opp = stones(40, 57), stones(1, 2, 16, 32)
        self.assertEqual(QUBIC.rules.find_vcf(opp, own), 0)
        plain, aware = Node(own, opp), Node(own, opp)
        search(uniform, [plain], 300, QUBIC)
        search(uniform, [aware], 300, QUBIC, vcf_budget=200)
        q = lambda r, c: r.W[r.moves.index(c)] / max(r.N[r.moves.index(c)], 1)
        losing = [c for c in aware.moves if QUBIC.rules.find_vcf(opp, own | 1 << c) is not None]
        self.assertTrue(losing)
        self.assertTrue(all(q(aware, c) <= -0.99 for c in losing if aware.N[aware.moves.index(c)] > 0))
        self.assertLess(sum(aware.N[aware.moves.index(c)] for c in losing) / aware.N.sum(),
                        sum(plain.N[plain.moves.index(c)] for c in losing) / plain.N.sum())

    def test_root_safety_removes_moves_that_allow_a_forced_win(self):
        own, opp = stones(40, 57), stones(1, 2, 16, 32)
        root = Node(own, opp)
        search(uniform, [root], 200, QUBIC, root_safety=300)
        allowed = [c for c in root.moves if QUBIC.rules.find_vcf(opp, own | 1 << c, 300) is None]
        self.assertEqual(sorted(root.moves), sorted(allowed))
        self.assertTrue(root.moves)
        self.assertEqual(int(root.N.sum()), root.visits)
        self.assertAlmostEqual(float(root.P.sum()), 1.0, places=5)

    def test_visits_reach_target_and_sum_matches(self):
        root = Node(0, 0)
        search(uniform, [root], 150, QUBIC)
        self.assertEqual(root.visits, 150)
        self.assertEqual(int(root.N.sum()), 150)
        self.assertAlmostEqual(float(visit_policy(root, QUBIC).sum()), 1.0, places=5)


class Bots(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_win_before_block(self):
        own, opp = stones(0, 1, 2), stones(16, 17, 18)
        for bot in (osp_bot, hsp_bot, tsp_bot):
            self.assertEqual(bot(own, opp, self.rng, QUBIC), 3)

    def test_block(self):
        own, opp = stones(40), stones(16, 17, 18)
        for bot in (osp_bot, hsp_bot, tsp_bot):
            self.assertEqual(bot(own, opp, self.rng, QUBIC), 19)

    def test_hsp_first_move_is_legal(self):
        for _ in range(300):
            self.assertIn(hsp_bot(0, 0, self.rng, QUBIC), range(QUBIC.cells))

    def test_vcf_finds_double_threat(self):
        self.assertEqual(QUBIC.rules.find_vcf(stones(1, 2, 16, 32), stones(5, 10, 25, 42)), 0)

    def test_bots_finish_games_on_every_board(self):
        for geo in ALL:
            for a, b in (("osp", "hsp"), ("hsp", "tsp"), ("random", "tsp")):
                score, played = play_match(BOTS[a], BOTS[b], 6, self.rng, geo)
                self.assertEqual(score["win"] + score["draw"] + score["loss"], 6, geo.name)
                for game in played:
                    self.assertEqual(len(set(game.moves)), len(game.moves))
                    if game.winner is not None:
                        self.assertTrue(geo.winning_line(game.moves))

    def test_tsp_beats_hsp(self):
        score, _ = play_match(tsp_bot, hsp_bot, 40, self.rng, QUBIC)
        self.assertGreater(score["win"], 30)


class ForcedWinProof(unittest.TestCase):
    """Партии заканчиваются по find_vcf, поэтому её ответ должен быть настоящим доказательством."""

    def assert_attacker_wins(self, geo, own, opp):
        cell = geo.rules.find_vcf(own, opp, budget=20000)
        self.assertIsNotNone(cell, "серия шахов оборвалась")
        self.assertFalse((own | opp) >> cell & 1, "ход в занятую клетку")
        mine = own | 1 << cell
        if geo.completes_line(mine, cell):
            return
        replies = geo.empty_cells(mine, opp)
        self.assertTrue(replies)
        for reply in replies:
            theirs = opp | 1 << reply
            self.assertFalse(geo.completes_line(theirs, reply), "соперник успел выиграть первым")
            if geo.winning_cells(mine, theirs):
                continue  # не закрыл угрозу: следующим ходом проигрывает
            self.assert_attacker_wins(geo, mine, theirs)

    def test_every_defence_loses(self):
        for geo, count in ((QUBIC, 80), (geometry(3, 3), 30), (geometry(5, 3), 15)):
            rng = np.random.default_rng(11)
            checked = 0
            while checked < count:
                own, opp = 0, 0
                for ply in range(int(rng.integers(geo.n, geo.cells * 2 // 5))):
                    free = geo.empty_cells(own, opp)
                    cell = free[int(rng.integers(len(free)))]
                    if geo.completes_line(own | 1 << cell, cell):
                        break
                    own, opp = opp, own | 1 << cell
                if geo.winning_cells(own, opp) or geo.winning_cells(opp, own):
                    continue
                if geo.rules.find_vcf(own, opp) is not None:
                    self.assert_attacker_wins(geo, own, opp)
                    checked += 1


class ShortLines(unittest.TestCase):
    """Плоские доски и большие поля, где линия короче стороны доски."""
    BOARDS = {(3, 2, None): 8, (4, 2, None): 10, (5, 2, 4): 28, (7, 2, 5): 60, (9, 2, 5): 140, (15, 2, 5): 572}

    def test_line_counts(self):
        for (n, d, k), count in self.BOARDS.items():
            geo = geometry(n, d, k)
            self.assertEqual(len(geo.lines), count, geo.name)
            self.assertTrue(all(len(line) == geo.k for line in geo.lines), geo.name)

    def test_lines_are_straight(self):
        for (n, d, k) in self.BOARDS:
            geo = geometry(n, d, k)
            for line in geo.lines:
                points = [(cell // n, cell % n) for cell in line]
                steps = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])}
                self.assertEqual(len(steps), 1, line)
                self.assertTrue(all(abs(v) <= 1 for v in steps.pop()), line)

    def test_names_and_defaults(self):
        self.assertEqual(geometry(15, 2, 5).name, "15x15k5")
        self.assertEqual(geometry(3, 2).name, "3x3")
        self.assertEqual(geometry(4, 3).k, 4)
        self.assertEqual(len(geometry(15, 2, 5).symmetries), 8)

    def test_symmetries_keep_lines(self):
        for (n, d, k) in self.BOARDS:
            geo = geometry(n, d, k)
            expected = {frozenset(line) for line in geo.lines}
            for perm in geo.symmetries:
                self.assertEqual({frozenset(int(perm[c]) for c in line) for line in geo.lines}, expected, geo.name)

    def test_gomoku_wins(self):
        geo = geometry(15, 2, 5)
        four = sum(1 << c for c in (32, 33, 34, 35))
        self.assertEqual(sorted(geo.winning_cells(four, 0)), [31, 36])
        self.assertFalse(geo.completes_line(four, 35))
        self.assertTrue(geo.completes_line(four | 1 << 36, 36))
        six = sum(1 << c for c in (32, 33, 34, 35, 36, 37))
        self.assertTrue(geo.completes_line(six, 35), "шесть в ряд тоже победа")
        # на краю поля пятая клетка есть только с одной стороны
        self.assertEqual(geo.winning_cells(sum(1 << c for c in (0, 1, 2, 3)), 0), [4])

    def test_bots_finish_games(self):
        rng = np.random.default_rng(5)
        for (n, d, k) in self.BOARDS:
            geo = geometry(n, d, k)
            for bot in BOTS.values():
                own, opp = 0, 0
                while True:
                    cell = bot(own, opp, rng, geo)
                    self.assertFalse((own | opp) >> cell & 1, geo.name)
                    mine = own | 1 << cell
                    if geo.completes_line(mine, cell) or mine | opp == geo.full:
                        break
                    own, opp = opp, mine


class FastRules(unittest.TestCase):
    """C-версия обязана совпадать с понятной версией на Python на досках всех размеров."""

    def test_vcf_agrees_with_python(self):
        for geo, count in ((QUBIC, 800), (geometry(3, 3), 500), (geometry(5, 3), 200), (geometry(4, 4), 60),
                           (geometry(5, 2, 4), 400), (geometry(7, 2, 5), 300), (geometry(15, 2, 5), 150)):
            found = 0
            for own, opp in random_positions(geo, count, 2):
                if geo.winning_cells(own, opp):
                    continue
                # без ограничения глубины, иначе ответ зависит от порядка перебора кандидатов
                c = geo.rules.find_vcf(own, opp, budget=30000, depth=64)
                py = find_vcf_py(own, opp, geo, budget=30000, depth=64)
                if (c is None) != (py is None):  # бюджет кончился: кто-то выбрал долгий путь первым
                    c = geo.rules.find_vcf(own, opp, budget=3000000, depth=64)
                    py = find_vcf_py(own, opp, geo, budget=300000, depth=64) if py is None else py
                self.assertEqual(c is None, py is None, (geo.name, own, opp))
                found += c is not None
            self.assertGreater(found, 0, geo.name)

    def test_analyse_codes(self):
        for geo in ALL:
            for own, opp in random_positions(geo, 300, 3):
                wins, threats = geo.winning_cells(own, opp), geo.winning_cells(opp, own)
                answer = geo.rules.analyse(own, opp, 0)
                code, cell = answer & 255, answer >> 8
                if wins:
                    self.assertEqual(code, fast.CODE_WIN)
                    self.assertIn(cell, wins)
                elif own | opp == geo.full:
                    self.assertEqual(code, fast.CODE_DRAW)
                elif len(threats) >= 2:
                    self.assertEqual(code, fast.CODE_LOST)
                    self.assertIn(cell, threats)
                elif threats:
                    self.assertEqual((code, cell), (fast.CODE_FORCED, threats[0]))
                else:
                    self.assertEqual(code, fast.CODE_NONE)


class Plateau(unittest.TestCase):
    def test_stops_only_after_patience_without_growth(self):
        from train import PlateauWatch
        watch = PlateauWatch(patience=3, min_delta=1.5, window=2)
        results = [watch.add(i, score) for i, score in enumerate([10, 20, 40, 60, 60, 60, 60, 60])]
        self.assertEqual(results[3], (True, False))   # среднее растёт, это новая лучшая версия
        self.assertEqual(watch.best_iter, 4)          # среднее 60 впервые на итерации с двумя 60
        self.assertEqual([r[1] for r in results], [False] * 7 + [True])

    def test_noise_does_not_count_as_growth(self):
        from train import PlateauWatch
        watch = PlateauWatch(patience=2, min_delta=5, window=1)
        watch.add(0, 50)
        self.assertEqual(watch.add(1, 53), (False, False))
        self.assertEqual(watch.add(2, 54), (False, True))

    def test_lowers_the_step_before_stopping(self):
        from train import PlateauWatch
        watch = PlateauWatch(patience=2, min_delta=5, window=1, drops=2)
        watch.add(0, 50)
        stops = [watch.add(i, 50)[1] for i in range(1, 7)]
        self.assertEqual(stops, [False] * 5 + [True])
        self.assertEqual(watch.drops_done, 2)

    def test_growth_after_a_lower_step_counts(self):
        from train import PlateauWatch
        watch = PlateauWatch(patience=2, min_delta=5, window=1, drops=1)
        watch.add(0, 50)
        watch.add(1, 50)
        watch.add(2, 50)
        self.assertEqual(watch.drops_done, 1)
        self.assertEqual(watch.add(3, 70), (True, False))
        self.assertEqual(watch.best_iter, 3)

    def test_never_stops_without_patience(self):
        from train import PlateauWatch
        watch = PlateauWatch(patience=0, window=1)
        self.assertFalse(any(watch.add(i, 50)[1] for i in range(20)))


class Cache(unittest.TestCase):
    def test_repeats_do_not_reach_the_net(self):
        asked = []

        def infer(owns, opps):
            asked.extend(zip(owns, opps))
            n = len(owns)
            values = np.array([o % 7 for o in owns], dtype=np.float32)
            return np.tile(np.arange(QUBIC.cells, dtype=np.float32), (n, 1)), values

        evaluator = CachedEvaluator(infer)
        priors, values = evaluator([1, 2, 1], [8, 9, 8])
        self.assertEqual(asked, [(1, 8), (2, 9)])
        self.assertEqual(values, [1.0, 2.0, 1.0])
        self.assertEqual(len(priors), 3)
        priors, values = evaluator([2, 3], [9, 10])
        self.assertEqual(asked, [(1, 8), (2, 9), (3, 10)])
        self.assertEqual(values, [2.0, 3.0])
        self.assertEqual((evaluator.requests, evaluator.computed), (5, 3))

    def test_limit_clears_and_answers_stay_correct(self):
        evaluator = CachedEvaluator(lambda owns, opps: (np.zeros((len(owns), QUBIC.cells)), np.array(owns, dtype=float)), limit=2)
        for i in range(10):
            self.assertEqual(evaluator([i], [0])[1], [float(i)])
        self.assertLessEqual(len(evaluator.cache), 3)


class Play(unittest.TestCase):
    def test_self_play_records_are_consistent(self):
        rng = np.random.default_rng(1)
        games, sims = self_play(uniform, 6, 3, 30, rng, QUBIC, early_finish=False)
        self.assertEqual(len(games), 6)
        self.assertGreater(sims, 0)
        for game in games:
            self.assertTrue(game.finished)
            self.assertEqual(len(set(game.moves)), len(game.moves))
            for ply, own, opp, pi in zip(game.plies, game.owns, game.opps, game.pis):
                # позиция совпадает с ходами партии и видна глазами того, чей ход
                x_stones = stones(*game.moves[:ply][0::2])
                o_stones = stones(*game.moves[:ply][1::2])
                self.assertEqual((own, opp), (x_stones, o_stones) if ply % 2 == 0 else (o_stones, x_stones))
                self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
                self.assertTrue(all(pi[c] == 0 for c in range(QUBIC.cells) if (own | opp) >> c & 1))
            if game.winner is not None:
                self.assertEqual(game.winner, (len(game.moves) - 1) % 2)
                self.assertTrue(QUBIC.winning_line(game.moves))

    def test_early_finish_records_the_proven_position(self):
        games, _ = self_play(uniform, 12, 4, 30, np.random.default_rng(3), QUBIC)
        ended_early = [g for g in games if g.adjudicated]
        self.assertTrue(ended_early)
        for game in ended_early:
            if not game.plies or game.plies[-1] != len(game.moves):
                continue  # двойная угроза: сеть там не вызывается, позиция не нужна
            self.assertEqual(game.winner, len(game.moves) % 2)
            cell = int(np.argmax(game.pis[-1]))
            self.assertEqual(game.pis[-1][cell], 1.0)
            self.assertEqual(QUBIC.rules.find_vcf(game.owns[-1], game.opps[-1]), cell)
            z = replay_arrays([game], QUBIC)["z"]
            self.assertEqual(z[-1], 1.0)

    def test_self_play_on_every_board(self):
        for geo in ALL:
            games, _ = self_play(uniform_net(geo), 4, 2, 10, np.random.default_rng(5), geo, vcf_budget=30)
            arrays = replay_arrays(games, geo)
            self.assertEqual(arrays["own"].shape, (len(arrays["z"]), geo.words), geo.name)
            self.assertEqual(arrays["pi"].shape[1], geo.cells)
            bits = geo.unpack(arrays["own"]) + geo.unpack(arrays["opp"])
            self.assertLessEqual(bits.max(), 1.0)

    def test_league_games_record_only_the_moves_of_the_network(self):
        games, _ = self_play(uniform, 12, 4, 20, np.random.default_rng(8), QUBIC, league=[(tsp_bot, 1.0)])
        self.assertTrue(all(g.bot is tsp_bot for g in games))
        self.assertEqual({g.bot_side for g in games}, {0, 1})
        for game in games:
            self.assertTrue(game.plies)
            for ply in game.plies:
                self.assertNotEqual(ply % 2, game.bot_side)
            arrays = replay_arrays([game], QUBIC)
            if game.winner is not None:
                self.assertEqual(arrays["z"][0], 1.0 if game.winner != game.bot_side else -1.0)

    def test_league_share_is_respected(self):
        games, _ = self_play(uniform, 40, 8, 5, np.random.default_rng(9), QUBIC, league=[(osp_bot, 0.5)])
        count = sum(g.bot is not None for g in games)
        self.assertTrue(10 <= count <= 30, count)

    def test_random_opening_is_not_recorded(self):
        games, _ = self_play(uniform, 20, 5, 20, np.random.default_rng(6), QUBIC, random_plies=4)
        openings = {g.opening for g in games}
        self.assertTrue(openings <= set(range(5)))
        self.assertGreater(len(openings), 1)
        for game in games:
            self.assertGreaterEqual(len(game.moves), game.opening)
            self.assertTrue(all(ply >= game.opening for ply in game.plies))

    def test_match_accounts_both_sides(self):
        agent = NetAgent(uniform, 20)
        score, played = play_match(agent, osp_bot, 8, np.random.default_rng(2), QUBIC)
        self.assertEqual(score["games"], 8)
        self.assertEqual(sum(g.a_first for g in played), 4)
        self.assertEqual(sum(score["as_first"]), 4)


if __name__ == "__main__":
    unittest.main()
