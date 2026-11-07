"""Сети: 3D-свёртки с остаточными блоками, сеть 2019 года и сеть на линиях для любого размера."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class LineFeatures(nn.Module):
    """Подсказка сети о линиях: для каждой клетки считает, сколько через неё проходит линий,
    где уже стоит k своих камней и нет чужих, и то же для соперника, для k от 1 до n-1."""

    def __init__(self, geo, persistent=True):
        super().__init__()
        self.n, self.shape = geo.k, geo.shape
        self.planes = 2 * (geo.k - 1)
        incidence = torch.zeros(len(geo.lines), geo.cells)
        for i, line in enumerate(geo.lines):
            incidence[i, line] = 1.0
        self.register_buffer("incidence", incidence, persistent=persistent)
        self.scale = float(geo.degree.max())

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            flat = x.float().flatten(2)
            own = flat[:, 0] @ self.incidence.T
            opp = flat[:, 1] @ self.incidence.T
            planes = []
            for k in range(1, self.n):
                planes.append(((own == k) & (opp == 0)).float() @ self.incidence)
                planes.append(((opp == k) & (own == 0)).float() @ self.incidence)
            extra = torch.stack(planes, dim=1).reshape((-1, self.planes) + self.shape) / self.scale
        return torch.cat((x, extra), dim=1)


class QubicNet(nn.Module):
    """Трёхмерные свёртки; подходит только для досок n×n×n."""

    def __init__(self, geo, channels=64, blocks=6, line_features=False):
        super().__init__()
        self.features = LineFeatures(geo) if line_features else nn.Identity()
        planes = 2 + (self.features.planes if line_features else 0)
        self.stem = nn.Sequential(nn.Conv3d(planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm3d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv3d(channels, 16, 1, bias=False), nn.BatchNorm3d(16),
                                    nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(16 * geo.cells, geo.cells))
        self.value = nn.Sequential(nn.Conv3d(channels, 8, 1, bias=False), nn.BatchNorm3d(8),
                                   nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(8 * geo.cells, 128),
                                   nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        x = self.blocks(self.stem(self.features(x)))
        return self.policy(x), self.value(x).squeeze(1)


class Net2019(nn.Module):
    """Сеть версии 2019 года (legacy/), перенесённая с Keras один в один.

    Вход одной плоскостью +1/-1/0, 4 свёртки 3×3×3, последняя без дополнения: куб 2×2×2.
    Затем слои 1024 и 512 с dropout. Настройки BatchNorm как в Keras. У политики 65 выходов,
    как в оригинале, но «пас» никогда не разрешён, поэтому наружу отдаются первые 64.
    """

    def __init__(self, channels=256, dropout=0.3):
        super().__init__()
        bn3 = lambda c: nn.BatchNorm3d(c, momentum=0.01, eps=1e-3)
        bn1 = lambda c: nn.BatchNorm1d(c, momentum=0.01, eps=1e-3)
        layers = []
        for i, padding in enumerate((1, 1, 1, 0)):
            layers += [nn.Conv3d(1 if i == 0 else channels, channels, 3, padding=padding, bias=False),
                       bn3(channels), nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(channels * 8, 1024, bias=False), bn1(1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                nn.Linear(1024, 512, bias=False), bn1(512), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.pi = nn.Linear(512, 65)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        board = (x[:, 0] - x[:, 1]).unsqueeze(1)
        s = self.fc(self.convs(board))
        return self.pi(s)[:, :64], torch.tanh(self.v(s)).squeeze(1)


class Links(nn.Module):
    """Какая клетка лежит на какой линии. Веса сети от этого не зависят, поэтому одну и ту же
    сеть можно загрузить на доску любого размера и даже другой размерности."""

    def __init__(self, geo):
        super().__init__()
        cells, lines = [], []
        for i, line in enumerate(geo.lines):
            cells += line
            lines += [i] * len(line)
        self.register_buffer("cell", torch.tensor(cells), persistent=False)
        self.register_buffer("line", torch.tensor(lines), persistent=False)
        self.register_buffer("cell_scale", torch.tensor(np.sqrt(geo.degree), dtype=torch.float32)[None, :, None], persistent=False)
        self.register_buffer("degree", torch.tensor(geo.degree / geo.degree.max(), dtype=torch.float32), persistent=False)
        self.n_cells, self.n_lines, self.line_scale = geo.cells, len(geo.lines), math.sqrt(geo.k)

    def to_lines(self, x):
        out = x.new_zeros(x.shape[0], self.n_lines, x.shape[2])
        return out.index_add_(1, self.line, x[:, self.cell]) / self.line_scale

    def to_cells(self, y):
        out = y.new_zeros(y.shape[0], self.n_cells, y.shape[2])
        return out.index_add_(1, self.cell, y[:, self.line]) / self.cell_scale


class LineBlock(nn.Module):
    """Клетки передают сообщения линиям, линии отвечают клеткам. Камни на одной линии связаны
    напрямую, как бы далеко на доске они ни стояли, и свёртки здесь не нужны."""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.to_line = nn.Linear(channels, channels)
        self.line1 = nn.Linear(channels, channels)
        self.line2 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ff1 = nn.Linear(channels, 2 * channels)
        self.ff2 = nn.Linear(2 * channels, channels)

    def forward(self, x, links):
        y = links.to_lines(self.to_line(self.norm1(x)))
        y = self.line2(F.relu(self.line1(F.relu(y))))
        x = x + links.to_cells(y)
        return x + self.ff2(F.relu(self.ff1(self.norm2(x))))


class LineNet(nn.Module):
    """Сеть на линиях: работает для любых n и d и сама соблюдает все симметрии игры."""

    def __init__(self, geo, channels=96, blocks=6):
        super().__init__()
        self.links = Links(geo)
        self.embed = nn.Linear(4, channels)
        self.blocks = nn.ModuleList([LineBlock(channels) for _ in range(blocks)])
        self.norm = nn.LayerNorm(channels)
        self.policy = nn.Linear(channels, 1)
        self.value1 = nn.Linear(2 * channels, channels)
        self.value2 = nn.Linear(channels, 1)

    def forward(self, x):
        flat = x.flatten(2).transpose(1, 2)  # (B, cells, 2): свои и чужие камни
        empty = 1 - flat.sum(dim=2, keepdim=True)
        degree = self.links.degree[None, :, None].expand(x.shape[0], -1, -1)
        h = self.embed(torch.cat((flat, empty, degree), dim=2))
        for block in self.blocks:
            h = block(h, self.links)
        h = self.norm(h)
        pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values), dim=1)
        return self.policy(h).squeeze(2), torch.tanh(self.value2(F.relu(self.value1(pooled)))).squeeze(1)


class LineLinks(Links):
    """Связи клеток и линий плюс то, что сеть должна знать о линии заранее: её тип.

    Тип линии — сколько координат меняется вдоль неё: ось, диагональ слоя, диагональ куба,
    диагональ гиперкуба. В 4D таких типов четыре, и ценность у них разная.
    """
    MAX_TYPES = 4

    def __init__(self, geo):
        super().__init__(geo)
        types = []
        for line in geo.lines:
            points = [geo_coords(cell, geo) for cell in line[:2]]
            types.append(sum(a != b for a, b in zip(*points)) - 1)
        self.register_buffer("types", torch.tensor(types), persistent=False)
        self.register_buffer("cell_sum_scale", torch.tensor(geo.degree / geo.degree.max(), dtype=torch.float32), persistent=False)
        self.n = geo.k
        self.max_degree = float(geo.degree.max())


def geo_coords(cell, geo):
    point = []
    for _ in range(geo.d):
        point.insert(0, cell % geo.n)
        cell //= geo.n
    return point


class LineNet2(nn.Module):
    """Сеть, собранная вокруг линий, для досок любого размера и размерности.

    Отличия от LineNet: сеть сразу видит состояние каждой линии (сколько камней осталось
    до победы каждой стороне, закрыта ли линия) и её тип, у линий есть своя память между блоками,
    а каждая клетка получает общий контекст доски. Веса не зависят от размера доски.
    """
    LINE_INPUTS = 11
    CELL_INPUTS = 11

    def __init__(self, geo, channels=96, blocks=6):
        super().__init__()
        self.links = LineLinks(geo)
        self.embed_cell = nn.Linear(self.CELL_INPUTS, channels)
        self.embed_line = nn.Linear(self.LINE_INPUTS, channels)
        self.line_type = nn.Embedding(LineLinks.MAX_TYPES, channels)
        self.blocks = nn.ModuleList([LineBlock2(channels) for _ in range(blocks)])
        self.norm = nn.LayerNorm(channels)
        self.policy = nn.Linear(channels, 1)
        self.value1 = nn.Linear(3 * channels, channels)
        self.value2 = nn.Linear(channels, 1)

    def inputs(self, x):
        links = self.links
        flat = x.flatten(2).float()  # (B, 2, cells)
        own, opp = flat[:, 0], flat[:, 1]
        own_count = own.new_zeros(own.shape[0], links.n_lines).index_add_(1, links.line, own[:, links.cell])
        opp_count = own.new_zeros(own.shape[0], links.n_lines).index_add_(1, links.line, opp[:, links.cell])
        n = float(links.n)
        open_own, open_opp = opp_count == 0, own_count == 0
        need = [(open_own & (own_count == n - k)).float() for k in (1, 2, 3)] + [(open_opp & (opp_count == n - k)).float() for k in (1, 2, 3)]
        line_inputs = torch.stack([own_count / n, opp_count / n, open_own.float(), open_opp.float(),
                                   (~open_own & ~open_opp).float()] + need, dim=2)
        # для клетки: сколько через неё проходит линий в каждом из этих состояний
        need_on_cells = torch.stack(need, dim=2)
        cell_need = need_on_cells.new_zeros(own.shape[0], links.n_cells, 6).index_add_(1, links.cell, need_on_cells[:, links.line]) / links.max_degree
        stage = (own.sum(dim=1, keepdim=True) + opp.sum(dim=1, keepdim=True)) / links.n_cells
        cell_inputs = torch.cat((own[..., None], opp[..., None], (1 - own - opp)[..., None],
                                 links.degree[None, :, None].expand(own.shape[0], -1, -1),
                                 stage[:, None].expand(-1, links.n_cells, 1), cell_need), dim=2)
        return cell_inputs, line_inputs

    def forward(self, x):
        links = self.links
        with torch.autocast(x.device.type, enabled=False):
            cell_inputs, line_inputs = self.inputs(x)
        h = self.embed_cell(cell_inputs)
        g = self.embed_line(line_inputs) + self.line_type(links.types)[None]
        for block in self.blocks:
            h, g = block(h, g, links)
        h = self.norm(h)
        pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values, g.mean(dim=1)), dim=1)
        return self.policy(h).squeeze(2), torch.tanh(self.value2(F.relu(self.value1(pooled)))).squeeze(1)


class LineBlock2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm_cell = nn.LayerNorm(channels)
        self.norm_line = nn.LayerNorm(channels)
        self.to_line = nn.Linear(channels, channels)
        self.line1 = nn.Linear(channels, channels)
        self.line2 = nn.Linear(channels, channels)
        self.to_cell = nn.Linear(channels, channels)
        self.board = nn.Linear(channels, channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff1 = nn.Linear(channels, 2 * channels)
        self.ff2 = nn.Linear(2 * channels, channels)

    def forward(self, h, g, links):
        n = self.norm_cell(h)
        y = self.norm_line(g) + links.to_lines(self.to_line(n))
        g = g + self.line2(F.relu(self.line1(F.relu(y))))
        h = h + links.to_cells(self.to_cell(g)) + self.board(n.mean(dim=1, keepdim=True))
        return h + self.ff2(F.relu(self.ff1(self.norm_ff(h)))), g


class ResBlock2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(y)), inplace=True)


class ConvNet2d(nn.Module):
    """Обычная свёрточная сеть для плоских досок: так устроены сильные программы для гомоку.

    Только свёртки, поэтому веса подходят к полю любого размера. Третья плоскость из единиц
    нужна, чтобы сеть видела края поля.
    """

    def __init__(self, channels, blocks):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock2d(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.value_conv = nn.Conv2d(channels, 16, 1)
        self.value = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        x = x.float()
        h = self.blocks(self.stem(torch.cat((x, torch.ones_like(x[:, :1])), dim=1)))
        v = F.relu(self.value_conv(h), inplace=True)
        pooled = torch.cat((v.mean(dim=(2, 3)), v.amax(dim=(2, 3))), dim=1)
        return self.policy(h).flatten(1), torch.tanh(self.value(pooled)).squeeze(1)


class Conv4d(nn.Module):
    """Четырёхмерная свёртка с ядром 3×3×3×3. В PyTorch такой нет, поэтому она собрана из трёхмерной.

    Такое ядро — это три ядра 3×3×3, по одному на шаг вдоль четвёртой оси, поэтому выход в слое w
    складывается из трёхмерных свёрток слоёв w-1, w и w+1. Весов 81 на пару каналов, как и положено.
    Вход и выход: (B, каналы, W, Z, Y, X). За краем доски нули.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, 3 * out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        batch, _, layers = x.shape[:3]
        y = self.conv(x.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch, layers))  # (B, W, 3·каналы, Z, Y, X)
        before, same, after = y.chunk(3, dim=2)
        empty = same[:, :1] * 0
        return (same + torch.cat((empty, before[:, :-1]), dim=1) + torch.cat((after[:, 1:], empty), dim=1)).transpose(1, 2)


class BatchNorm4d(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.flatten(2)).view(x.shape)


class ResBlock4d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv4d(channels, channels)
        self.bn1 = BatchNorm4d(channels)
        self.conv2 = Conv4d(channels, channels)
        self.bn2 = BatchNorm4d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(y)), inplace=True)


class ConvNet4d(nn.Module):
    """Свёрточная сеть для доски в четырёх измерениях. Плоскость из единиц показывает края поля.

    line_features добавляет те же подсказки о линиях, что были у лучшей трёхмерной свёрточной сети.
    """

    def __init__(self, geo, channels, blocks, line_features=False):
        super().__init__()
        self.features = LineFeatures(geo) if line_features else None
        planes = 3 + (self.features.planes if line_features else 0)
        self.stem = nn.Sequential(Conv4d(planes, channels), BatchNorm4d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock4d(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv1d(channels, 16, 1, bias=False), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Conv1d(16, 1, 1))
        self.value_conv = nn.Conv1d(channels, 16, 1)
        self.value = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        x = x.float()
        planes = self.features(x) if self.features is not None else x
        h = self.blocks(self.stem(torch.cat((planes, torch.ones_like(x[:, :1])), dim=1))).flatten(2)
        v = F.relu(self.value_conv(h), inplace=True)
        pooled = torch.cat((v.mean(dim=2), v.amax(dim=2)), dim=1)
        return self.policy(h).squeeze(1), torch.tanh(self.value(pooled)).squeeze(1)


def board_conv(d, in_channels, out_channels):
    if d == 2:
        return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    if d == 3:
        return nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
    return Conv4d(in_channels, out_channels)


class HybridBlock(nn.Module):
    """Свёртки видят узор вокруг клетки, сообщения по линиям сразу связывают камни одной линии."""

    def __init__(self, d, channels):
        super().__init__()
        self.conv1 = board_conv(d, channels, channels)
        self.bn1 = BatchNorm4d(channels)
        self.conv2 = board_conv(d, channels, channels)
        self.bn2 = BatchNorm4d(channels)
        self.lines = LineBlock(channels)

    def forward(self, x, links):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(x + self.bn2(self.conv2(y)), inplace=True)
        z = self.lines(y.flatten(2).transpose(1, 2), links)
        return z.transpose(1, 2).view(y.shape)


class HybridNet(nn.Module):
    """Гибрид для плоских, объёмных и четырёхмерных досок. Веса подходят к доскам той же размерности."""

    def __init__(self, geo, channels, blocks):
        super().__init__()
        self.links = Links(geo)
        self.features = LineFeatures(geo, persistent=False)  # таблица линий зависит от доски, это не веса
        planes = 3 + self.features.planes
        self.stem = nn.Sequential(board_conv(geo.d, planes, channels), BatchNorm4d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(HybridBlock(geo.d, channels) for _ in range(blocks))
        self.policy = nn.Sequential(nn.Conv1d(channels, 16, 1, bias=False), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Conv1d(16, 1, 1))
        self.value_conv = nn.Conv1d(channels, 16, 1)
        self.value = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        x = x.float()
        h = self.stem(torch.cat((self.features(x), torch.ones_like(x[:, :1])), dim=1))
        for block in self.blocks:
            h = block(h, self.links)
        h = h.flatten(2)
        v = F.relu(self.value_conv(h), inplace=True)
        pooled = torch.cat((v.mean(dim=2), v.amax(dim=2)), dim=1)
        return self.policy(h).squeeze(1), torch.tanh(self.value(pooled)).squeeze(1)


def build_net(arch, geo, channels, blocks):
    if arch == "hybrid":
        return HybridNet(geo, channels, blocks)
    if arch in ("conv4d", "conv4d_lines"):
        if geo.d != 4:
            raise ValueError("conv4d только для четырёхмерной доски")
        return ConvNet4d(geo, channels, blocks, line_features=arch == "conv4d_lines")
    if arch == "conv2d":
        if geo.d != 2:
            raise ValueError("conv2d только для плоских досок")
        return ConvNet2d(channels, blocks)
    if arch == "linenet2":
        return LineNet2(geo, channels, blocks)
    if arch == "linenet":
        return LineNet(geo, channels, blocks)
    if arch == "net2019":
        return Net2019(channels)
    return QubicNet(geo, channels, blocks, line_features=arch == "lines")


def occupied_mask(x):
    """Занятые клетки по входу сети: на них ходить нельзя, вероятность обнуляется."""
    return x.sum(dim=1).flatten(1) > 0


class Evaluator:
    """Считает пачку позиций на видеокарте и отдаёт массивы numpy для поиска."""

    def __init__(self, net, device, geo, views=1):
        self.net = net
        self.device = device
        self.geo = geo
        self.positions = 0
        # views > 1: сеть смотрит на позицию после нескольких поворотов доски, ответы усредняются
        picks = np.linspace(0, len(geo.symmetries) - 1, views).astype(int)
        self.perms = torch.from_numpy(geo.symmetries[picks]).to(device) if views > 1 else None
        self.inverse = torch.argsort(self.perms, dim=1) if views > 1 else None

    @torch.inference_mode()
    def __call__(self, owns, opps):
        self.positions += len(owns)
        encoded = self.geo.encode_packed(owns, opps) if isinstance(owns, np.ndarray) else self.geo.encode(owns, opps)
        x = torch.from_numpy(encoded).to(self.device, non_blocking=True)
        if self.perms is not None:
            return self._many_views(x)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits, values = self.net(x)
        probs = F.softmax(logits.float().masked_fill(occupied_mask(x), float("-inf")), dim=1)
        return probs.cpu().numpy(), values.float().cpu().numpy()

    def _many_views(self, x):
        batch, views = len(x), len(self.perms)
        flat = x.flatten(2)
        turned = torch.cat([flat[:, :, perm] for perm in self.perms]).view((batch * views,) + x.shape[1:])
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits, values = self.net(turned)
        probs = F.softmax(logits.float().masked_fill(occupied_mask(turned), float("-inf")), dim=1).view(views, batch, -1)
        average = sum(probs[i][:, self.inverse[i]] for i in range(views)) / views
        return average.cpu().numpy(), values.float().view(views, batch).mean(dim=0).cpu().numpy()
