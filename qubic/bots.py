"""Боты-соперники. osp и hsp повторяют игроков из версии 2019 года, tsp заметно сильнее.

Каждый бот: функция (own, opp, rng, geo) -> номер клетки.
"""


def _pick(cells, rng):
    return cells[int(rng.integers(len(cells)))]


def random_bot(own, opp, rng, geo):
    return _pick(geo.empty_cells(own, opp), rng)


def osp_bot(own, opp, rng, geo):
    """osp: выигрывает, если можно; мешает победе соперника; иначе ходит случайно."""
    return _pick(geo.winning_cells(own, opp) or geo.winning_cells(opp, own) or geo.empty_cells(own, opp), rng)


def hsp_bot(own, opp, rng, geo):
    """hsp: как osp, но вместо случайного хода выбирает клетку, на линиях через которую
    уже больше всего камней. В 2019 году этот бот мог вернуть занятую клетку или «пас»;
    здесь выбор только из свободных."""
    forced = geo.winning_cells(own, opp) or geo.winning_cells(opp, own)
    if forced:
        return _pick(forced, rng)
    stones = own | opp
    scored = [(sum((m & stones).bit_count() for m in geo.lines_through[c]), c) for c in geo.empty_cells(own, opp)]
    best = max(score for score, _ in scored)
    return _pick([c for score, c in scored if score == best], rng)


def find_vcf_py(own, opp, geo, budget=400, depth=16):
    """Поиск серии шахов на Python. В работе используется версия из fast.c, эта понятнее
    и нужна тестам для сверки: каждый ход создаёт угрозу, ответ соперника вынужден.

    Возвращает первый ход такой серии или None.
    """
    left = [budget]
    seen = set()
    order = {c: -int(geo.degree[c]) for c in range(geo.cells)}

    def search(own, opp, depth):
        left[0] -= 1
        if left[0] < 0:
            return None
        wins = geo.winning_cells(own, opp)
        if wins:
            return wins[0]
        threats = geo.winning_cells(opp, own)
        if len(threats) >= 2 or depth == 0:
            return None
        candidates = set()
        for mask in geo.line_masks:
            if not mask & opp and (mask & own).bit_count() == geo.k - 2:
                rest = mask & ~own
                while rest:
                    low = rest & -rest
                    candidates.add(low.bit_length() - 1)
                    rest ^= low
        if threats:
            candidates &= set(threats)
        for cell in sorted(candidates, key=order.get):
            mine = own | 1 << cell
            replies = geo.winning_cells(mine, opp)
            if len(replies) >= 2:
                return cell
            if not replies:
                continue
            theirs = opp | 1 << replies[0]
            if (mine, theirs) in seen:
                continue
            seen.add((mine, theirs))
            if search(mine, theirs, depth - 1) is not None:
                return cell
        return None

    return search(own, opp, depth)


def _line_score(cell, own, opp, geo):
    score = 0
    for mask in geo.lines_through[cell]:
        mine = (mask & own).bit_count()
        theirs = (mask & opp).bit_count()
        if mine and theirs:
            continue
        score += 4 ** mine if not theirs else 3 * 4 ** (theirs - 1)
    return score


def tsp_bot(own, opp, rng, geo):
    """Бот с поиском серий шахов: находит свои форсированные победы и мешает чужим."""
    forced = geo.winning_cells(own, opp) or geo.winning_cells(opp, own)
    if forced:
        return _pick(forced, rng)
    attack = geo.rules.find_vcf(own, opp)
    if attack is not None:
        return attack
    cells = geo.empty_cells(own, opp)
    ranked = sorted(cells, key=lambda c: (_line_score(c, own, opp, geo), rng.random()), reverse=True)
    danger = geo.rules.find_vcf(opp, own)
    candidates = ranked[:10] if danger is not None else ranked[:3]
    for cell in candidates:
        if geo.rules.find_vcf(opp, own | 1 << cell, budget=150) is None:
            return cell
    return danger if danger is not None else ranked[0]


BOTS = {"osp": osp_bot, "hsp": hsp_bot, "tsp": tsp_bot, "random": random_bot}
