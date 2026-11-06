"""Запуск: python -m unittest discover -s tests -v"""
import itertools
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qubic.game import geometry
from qubic.net import Conv4d, LineFeatures, LineNet, LineNet2, Net2019, QubicNet, build_net

QUBIC = geometry(4, 3)


class Nets(unittest.TestCase):
    def test_net2019_has_the_size_of_the_keras_original(self):
        # Keras: conv 27c и 3×27c², 4 BN по 2c, Dense 8c×1024 и 1024×512 без bias,
        # BN 2×1024 и 2×512, pi 512×65+65, v 512+1; необучаемые средние BN в это не входят
        c = 256
        expected = 27 * c + 3 * 27 * c * c + 4 * 2 * c + 8 * c * 1024 + 1024 * 512 + 2 * 1024 + 2 * 512 + 512 * 65 + 65 + 513
        self.assertEqual(sum(p.numel() for p in Net2019(c).parameters()), expected)
        self.assertEqual(expected, 7975746)

    def test_outputs(self):
        x = torch.from_numpy(QUBIC.encode([1, 2, 4], [8, 16, 32]))
        for net in (Net2019(16), QubicNet(QUBIC, 8, 1), QubicNet(QUBIC, 8, 1, line_features=True), LineNet(QUBIC, 8, 1)):
            logits, values = net.eval()(x)
            self.assertEqual(logits.shape, (3, QUBIC.cells))
            self.assertEqual(values.shape, (3,))
            self.assertTrue(((values >= -1) & (values <= 1)).all())

    def test_line_features_count_open_lines(self):
        # у своих камней 0,1 в линии 0-1-2-3: клетки 2 и 3 получают по такой линии «двух своих»
        own, opp = (1 << 0) | (1 << 1), 1 << 63
        features = LineFeatures(QUBIC)
        out = features(torch.from_numpy(QUBIC.encode([own], [opp])))
        self.assertEqual(out.shape, (1, 8, 4, 4, 4))
        two_own = out[0, 2 + 2].flatten() * features.scale
        self.assertEqual(two_own[2].item(), 1.0)
        self.assertEqual(two_own[3].item(), 1.0)
        self.assertEqual(two_own[60].item(), 0.0)
        one_opp = out[0, 2 + 1].flatten() * features.scale
        self.assertEqual(one_opp[62].item(), 1.0)

    def test_line_features_respect_symmetry(self):
        own, opp = (1 << 0) | (1 << 21), (1 << 5) | (1 << 42)
        features = LineFeatures(QUBIC)
        x = torch.from_numpy(QUBIC.encode([own], [opp]))
        perm = torch.from_numpy(QUBIC.symmetries[17])
        rotated = x.flatten(2)[:, :, perm].reshape(1, 2, 4, 4, 4)
        self.assertTrue(torch.equal(features(rotated).flatten(2), features(x).flatten(2)[:, :, perm]))

    def test_net2019_sees_the_board_as_plus_and_minus(self):
        net = Net2019(8).eval()
        seen = {}
        net.convs[0].register_forward_hook(lambda m, inp, out: seen.update(board=inp[0]))
        net(torch.from_numpy(QUBIC.encode([1], [2])))
        self.assertEqual(seen["board"].shape, (1, 1, 4, 4, 4))
        self.assertEqual(seen["board"][0, 0, 0, 0, 0].item(), 1.0)
        self.assertEqual(seen["board"][0, 0, 0, 0, 1].item(), -1.0)

    def test_linenet_keeps_every_symmetry_by_itself(self):
        for geo in (QUBIC, geometry(4, 4), geometry(3, 3)):
            net = LineNet(geo, 16, 2).eval()
            own, opp = (1 << 0) | (1 << 5), (1 << 7) | (1 << 12)
            x = torch.from_numpy(geo.encode([own], [opp]))
            perm = torch.from_numpy(geo.symmetries[len(geo.symmetries) // 3])
            rotated = x.flatten(2)[:, :, perm].reshape((1, 2) + geo.shape)
            with torch.no_grad():
                logits, value = net(x)
                logits2, value2 = net(rotated)
            self.assertTrue(torch.allclose(logits2, logits[:, perm], atol=1e-4), geo.name)
            self.assertAlmostEqual(value.item(), value2.item(), places=4)

    def test_linenet2_keeps_every_symmetry_and_fits_every_board(self):
        small = LineNet2(geometry(3, 3), 16, 2)
        for geo in (geometry(3, 3), QUBIC, geometry(5, 3), geometry(4, 4)):
            net = LineNet2(geo, 16, 2).eval()
            net.load_state_dict(small.state_dict())
            own, opp = (1 << 0) | (1 << 5), (1 << 7) | (1 << 12)
            x = torch.from_numpy(geo.encode([own], [opp]))
            perm = torch.from_numpy(geo.symmetries[len(geo.symmetries) // 3])
            rotated = x.flatten(2)[:, :, perm].reshape((1, 2) + geo.shape)
            with torch.no_grad():
                logits, value = net(x)
                logits2, value2 = net(rotated)
            self.assertEqual(logits.shape, (1, geo.cells))
            self.assertTrue(torch.allclose(logits2, logits[:, perm], atol=1e-4), geo.name)
            self.assertAlmostEqual(value.item(), value2.item(), places=4)

    def test_linenet2_sees_threats_directly(self):
        # три своих камня на линии: клетка 3 получает отметку «остался 1 камень» сразу, без обучения
        net = LineNet2(QUBIC, 8, 1).eval()
        x = torch.from_numpy(QUBIC.encode([(1 << 0) | (1 << 1) | (1 << 2)], [1 << 63]))
        cell_inputs, line_inputs = net.inputs(x)
        self.assertGreater(cell_inputs[0, 3, 5].item(), 0)  # need_own[1] стоит шестой
        self.assertEqual(cell_inputs[0, 40, 5].item(), 0)
        self.assertEqual(line_inputs.shape[2], LineNet2.LINE_INPUTS)
        self.assertEqual(cell_inputs.shape[2], LineNet2.CELL_INPUTS)

    def test_line_types_on_the_4d_board(self):
        geo = geometry(4, 4)
        net = LineNet2(geo, 8, 1)
        counts = torch.bincount(net.links.types).tolist()
        self.assertEqual(sum(counts), 520)
        self.assertEqual(counts[0], 256)   # оси: 4 направления по 64 линии
        self.assertEqual(counts[3], 8)     # диагонали гиперкуба

    def test_linenet_weights_fit_every_board(self):
        small = LineNet(geometry(3, 3), 16, 2)
        for geo in (QUBIC, geometry(5, 3), geometry(4, 4)):
            other = LineNet(geo, 16, 2).eval()
            other.load_state_dict(small.state_dict())
            logits, values = other(torch.from_numpy(geo.encode([1], [2])))
            self.assertEqual(logits.shape, (1, geo.cells))


class FourDimensionalConvolution(unittest.TestCase):
    """В PyTorch нет Conv4d: собранная вручную свёртка обязана совпасть с определением."""

    def test_matches_the_definition(self):
        torch.manual_seed(1)
        conv = Conv4d(2, 3)
        x = torch.randn(2, 2, 4, 4, 4, 4)
        fast = conv(x)
        kernel = conv.conv.weight.detach().reshape(3, 3, 2, 3, 3, 3)  # шаг по w, выходной канал, входной, ядро 3×3×3
        slow = torch.zeros_like(fast)
        for out in range(3):
            for channel in range(2):
                for point in itertools.product(range(4), repeat=4):
                    for shift in itertools.product((-1, 0, 1), repeat=4):
                        source = [p + s for p, s in zip(point, shift)]
                        if all(0 <= c < 4 for c in source):
                            weight = kernel[shift[0] + 1, out, channel, shift[1] + 1, shift[2] + 1, shift[3] + 1]
                            slow[(slice(None), out) + point] += weight * x[(slice(None), channel) + tuple(source)]
        self.assertLess((fast - slow).abs().max().item(), 1e-4)

    def test_weight_count(self):
        self.assertEqual(sum(p.numel() for p in Conv4d(5, 7).parameters()), 81 * 5 * 7)

    def test_moves_with_the_input_along_every_axis(self):
        torch.manual_seed(2)
        conv = Conv4d(1, 2)
        x = torch.zeros(1, 1, 6, 6, 6, 6)
        x[0, 0, 2, 2, 2, 2] = 1
        base = conv(x)
        for axis in range(4):
            moved = conv(torch.roll(x, 1, dims=axis + 2))
            self.assertLess((torch.roll(base, 1, dims=axis + 2) - moved).abs().max().item(), 1e-5, axis)

    def test_two_layers_reach_a_whole_diagonal(self):
        # линия из четырёх клеток укладывается в поле зрения двух слоёв и по главной диагонали
        conv = nn.Sequential(Conv4d(1, 2), Conv4d(2, 1))
        x = torch.zeros(1, 1, 4, 4, 4, 4, requires_grad=True)
        conv(x)[0, 0, 3, 3, 3, 3].backward()
        self.assertNotEqual(x.grad[0, 0, 1, 1, 1, 1].item(), 0)

    def test_networks(self):
        geo = geometry(4, 4)
        for arch in ("conv4d", "conv4d_lines"):
            net = build_net(arch, geo, 8, 2).eval()
            policy, value = net(torch.from_numpy(geo.encode([1], [2])))
            self.assertEqual(policy.shape, (1, 256), arch)
            self.assertEqual(value.shape, (1,), arch)
        with self.assertRaises(ValueError):
            build_net("conv4d", QUBIC, 8, 2)


class Hybrid(unittest.TestCase):
    def test_fits_every_dimension(self):
        for geo in (geometry(3, 2), geometry(15, 2, 5), QUBIC, geometry(4, 4)):
            net = build_net("hybrid", geo, 8, 2).eval()
            policy, value = net(torch.from_numpy(geo.encode([1], [2])))
            self.assertEqual(policy.shape, (1, geo.cells), geo.name)
            self.assertEqual(value.shape, (1,), geo.name)

    def test_weights_move_between_boards_of_one_dimension(self):
        small = build_net("hybrid", geometry(9, 2, 5), 8, 2)
        build_net("hybrid", geometry(15, 2, 5), 8, 2).load_state_dict(small.state_dict())

    def test_a_stone_reaches_the_far_end_of_a_line_at_once(self):
        # один блок: свёртка ядром 3 такую даль не достаёт, а сообщение по линии достаёт
        geo = geometry(15, 2, 5)
        net = build_net("hybrid", geo, 8, 1).eval()
        x = torch.zeros((1, 2) + geo.shape, requires_grad=True)
        net(x)[0][0, 7 * 15 + 11].backward()
        self.assertNotEqual(x.grad[0, 0, 7, 7].item(), 0)


class ShortLineBoards(unittest.TestCase):
    """Плоские доски и линии короче стороны доски."""

    def test_weights_fit_any_board(self):
        small = LineNet(geometry(5, 2, 4), 16, 2)
        for geo in (geometry(3, 2), geometry(7, 2, 5), geometry(15, 2, 5)):
            net = LineNet(geo, 16, 2)
            net.load_state_dict(small.state_dict())
            policy, value = net(torch.zeros((2, 2) + geo.shape))
            self.assertEqual(policy.shape, (2, geo.cells))
            self.assertEqual(value.shape[0], 2)

    def test_convolutions_fit_any_flat_board(self):
        small = build_net("conv2d", geometry(7, 2, 5), 16, 2)
        for geo in (geometry(3, 2), geometry(15, 2, 5)):
            net = build_net("conv2d", geo, 16, 2).eval()
            net.load_state_dict(small.state_dict())
            policy, value = net(torch.zeros((2, 2) + geo.shape))
            self.assertEqual(policy.shape, (2, geo.cells))
            self.assertEqual(value.shape, (2,))
        with self.assertRaises(ValueError):
            build_net("conv2d", geometry(4, 3), 16, 2)

    def test_second_network_sees_line_length(self):
        geo = geometry(7, 2, 5)
        net = LineNet2(geo, 16, 2)
        own = torch.zeros((1,) + geo.shape)
        for cell in (22, 23, 24, 25):  # четыре в ряд в середине строки: пятый камень можно поставить с двух сторон
            own[0, cell // 7, cell % 7] = 1
        cells, lines = net.inputs(torch.stack((own, torch.zeros_like(own)), dim=1))
        self.assertEqual(lines[0, :, 5].sum().item(), 2, "линии, где не хватает одного камня")

    def test_exact_results(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
        from solve_small import Solver
        self.assertEqual(Solver(geometry(3, 2)).value(0, 0), 0, "крестики-нолики 3×3: ничья")
        self.assertEqual(Solver(geometry(4, 2, 3)).value(0, 0), 1, "три в ряд на 4×4: выигрывает первый")


if __name__ == "__main__":
    unittest.main()
