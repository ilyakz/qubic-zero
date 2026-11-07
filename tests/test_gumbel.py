import unittest

import numpy as np

from qubic.game import geometry
from qubic.mcts import GumbelPlan, Node, gumbel_search
from qubic.play import self_play


def uniform(geo, favourite=None):
    """Сеть-заглушка: все ходы равны, оценка 0; favourite — ход, который ей нравится больше других."""
    def infer(owns, opps):
        p = np.full((len(owns), geo.cells), 1.0 / geo.cells, dtype=np.float32)
        if favourite is not None:
            p[:, favourite] = 50.0 / geo.cells
        return p, np.zeros(len(owns), dtype=np.float32)
    return infer


def fresh_roots(geo, count=1):
    roots = [Node(0, 0) for _ in range(count)]
    for root in roots:
        root.analyse(geo)
    return roots


class TestGumbel(unittest.TestCase):
    def setUp(self):
        self.geo = geometry(4, 3)
        self.rng = np.random.default_rng(1)

    def test_simulations_add_up_and_policy_is_a_distribution(self):
        roots = fresh_roots(self.geo, 3)
        plans, done = gumbel_search(uniform(self.geo), roots, 40, self.geo, 8, [1.0] * 3, self.rng)
        self.assertEqual(done, 120)
        for plan in plans:
            self.assertEqual(plan.root.visits, 40)
            self.assertEqual(int((plan.root.N > 0).sum()), 8)
            pi = plan.policy(self.geo)
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
            self.assertIn(plan.best_move(), plan.root.moves)

    def test_halving_spends_more_on_survivors(self):
        root = fresh_roots(self.geo)[0]
        plans, _ = gumbel_search(uniform(self.geo), [root], 64, self.geo, 8, [1.0], self.rng)
        visits = np.sort(root.N[root.N > 0])
        self.assertGreater(visits[-1], 2 * visits[0])

    def test_target_prefers_the_move_that_search_found_good(self):
        geo = self.geo
        root = fresh_roots(geo)[0]
        root.expand(uniform(geo)([0], [0])[0][0], 0.0, geo)
        plan = GumbelPlan(root, 16, 4, self.rng, 0.0)
        root.N[0], root.W[0] = 4, 3.6
        root.N[1], root.W[1] = 4, -3.6
        pi = plan.policy(geo)
        self.assertGreater(pi[root.moves[0]], 10 * pi[root.moves[2]])
        self.assertGreater(pi[root.moves[2]], 10 * pi[root.moves[1]])

    def test_network_favourite_is_tried_without_noise(self):
        root = fresh_roots(self.geo)[0]
        plans, _ = gumbel_search(uniform(self.geo, favourite=21), [root], 8, self.geo, 2, [0.0], self.rng)
        self.assertGreater(root.N[root.moves.index(21)], 0)

    def test_self_play_uses_it(self):
        games, sims = self_play(uniform(self.geo), 4, 2, 16, self.rng, self.geo, gumbel_width=4)
        self.assertEqual(len(games), 4)
        self.assertGreater(sims, 0)
        for game in games:
            for pi in game.pis:
                self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
