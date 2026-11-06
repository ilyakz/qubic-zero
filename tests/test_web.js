// Сверка движка на JavaScript с версией на Python: node tests/test_web.js
// Нужны файлы, которые пишет: python tools/export_web.py <запуск> --fixtures tests/web_fixtures_<доска>.json
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const Engine = require('../docs/engine.js');

const lines = { '3,3': 49, '4,3': 76, '5,3': 109, '4,4': 520, '3,2': 8, '4,2': 10, '5,2,4': 28, '7,2,5': 60, '9,2,5': 140, '15,2,5': 572 };
for (const key of Object.keys(lines)) {
  const [n, d, k] = key.split(',').map(Number);
  const geo = Engine.geometry(n, d, k);
  assert.strictEqual(geo.lines.length, lines[key], `lines of ${geo.name}`);
}

// Ход и отмена возвращают всё как было
const geo = Engine.geometry(4, 3);
const pos = new Engine.Position(geo, [0, 21, 1]);
const before = [pos.key, pos.side, pos.ply, Array.from(pos.count[0]).join()].join('|');
pos.play(5); pos.undo(5);
assert.strictEqual([pos.key, pos.side, pos.ply, Array.from(pos.count[0]).join()].join('|'), before, 'undo');

// Победа и линия определяются верно
const won = Engine.describe(geo, [0, 16, 1, 17, 2, 18, 3]);
assert.strictEqual(won.winner, 0);
assert.deepStrictEqual(Array.from(won.line), [0, 1, 2, 3]);
assert.strictEqual(Engine.describe(geo, []).finished, false);

// В гомоку побеждают пять в ряд и больше, а четырёх мало
{
  const g = Engine.geometry(15, 2, 5);
  assert.strictEqual(g.name, '15x15k5');
  const row = [32, 33, 34, 35], other = [100, 102, 104, 106];
  const moves = [];
  row.forEach((c, i) => moves.push(c, other[i]));
  assert.strictEqual(Engine.describe(g, moves).finished, false, 'four in a row is not a win');
  assert.deepStrictEqual(Engine.describe(g, moves).threats[0].sort((a, b) => a - b), [31, 36]);
  assert.strictEqual(Engine.describe(g, [...moves, 36]).winner, 0);
  const six = Engine.describe(g, [32, 100, 33, 102, 34, 104, 36, 106, 37, 108, 35]);
  assert.strictEqual(six.winner, 0, 'six in a row counts');
}

// Двойная угроза находится
const double = new Engine.Position(geo);
for (const [x, o] of [[1, 5], [2, 10], [16, 25], [32, 42]]) { double.play(x); double.play(o); }
assert.strictEqual(Engine.findVcf(double), 0, 'double threat');

// Боты всегда делают допустимый ход и доигрывают партию на всех досках
for (const [n, d, k] of [[3, 3], [4, 3], [5, 3], [4, 4], [3, 2], [4, 2], [5, 2, 4], [7, 2, 5], [15, 2, 5]]) {
  const g = Engine.geometry(n, d, k);
  for (const name of Object.keys(Engine.bots)) {
    const moves = [];
    const p = new Engine.Position(g);
    let seed = 7;
    const rng = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    while (!Engine.describe(g, moves).finished) {
      const cell = Engine.bots[name](p, rng);
      assert.strictEqual(p.board[cell], 0, `${name} played on an occupied cell`);
      p.play(cell);
      moves.push(cell);
    }
  }
}

// Поиск без сети берёт выигрыш и закрывает угрозу
const uniform = new Engine.UniformNet(geo);
assert.strictEqual(Engine.netMove(new Engine.Position(geo, [0, 16, 1, 17, 2, 18]), uniform, 50).cell, 3);
assert.strictEqual(Engine.netMove(new Engine.Position(geo, [0, 16, 1, 17, 20, 18]), uniform, 50).cell, 19);

// Сверка с ответами сети на Python
let compared = 0;
for (const file of fs.readdirSync(__dirname).filter(f => f.startsWith('web_fixtures') && f.endsWith('.json'))) {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, file)));
  const g = Engine.geometry(data.n, data.d, data.k);
  assert.strictEqual(g.lines.length, data.lines);
  const dir = path.join(__dirname, '..', 'docs', 'models');
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, g.name + '.json')));
  const raw = fs.readFileSync(path.join(dir, g.name + '.bin'));
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const net = Engine.makeNet(g, manifest, buffer);
  let worstPrior = 0, worstValue = 0, vcfAgree = 0, vcfTotal = 0;
  for (const c of data.cases) {
    const p = new Engine.Position(g, c.moves);
    const { priors, value } = net.evaluate(p);
    worstValue = Math.max(worstValue, Math.abs(value - c.value));
    for (let i = 0; i < g.cells; i++) worstPrior = Math.max(worstPrior, Math.abs(priors[i] - c.priors[i]));
    assert.deepStrictEqual(p.wins(p.side).sort((a, b) => a - b), c.wins, 'wins');
    assert.deepStrictEqual(p.wins(1 - p.side).sort((a, b) => a - b), c.threats, 'threats');
    if (c.vcf !== null) {
      vcfTotal++;
      vcfAgree += (Engine.findVcf(p, 3000, 64) >= 0) === c.vcf ? 1 : 0;
    }
    compared++;
  }
  console.log(`${g.name}: наибольшее расхождение с PyTorch: вероятность ${worstPrior.toExponential(2)}, оценка ${worstValue.toExponential(2)}; серии шахов совпали ${vcfAgree}/${vcfTotal}`);
  assert(worstPrior < 2e-3, 'priors differ');
  assert(worstValue < 2e-3, 'value differs');
  assert(vcfAgree >= vcfTotal - 1, 'threat sequences differ');

  // Сколько занимает одна оценка в JS
  const start = Date.now();
  for (let i = 0; i < 20; i++) net.evaluate(new Engine.Position(g, data.cases[i % data.cases.length].moves));
  console.log(`  одна оценка в JavaScript: ${((Date.now() - start) / 20).toFixed(1)} мс`);
}
console.log(`Все проверки прошли, сверено позиций с сетью: ${compared}`);

// Проверка в корне убирает ходы, отдающие серию шахов
{
  const g = Engine.geometry(4, 3);
  const p = new Engine.Position(g);
  for (const c of [40, 1, 57, 2, 11, 16, 13, 32]) p.play(c);
  const safe = Engine.search(p, new Engine.UniformNet(g), 80, { rootSafety: 300 });
  for (const c of safe.moves) {
    p.play(c);
    assert.strictEqual(Engine.findVcf(p, 300), -1, 'an unsafe move stayed at the root');
    p.undo(c);
  }
  const plain = Engine.search(p, new Engine.UniformNet(g), 80, {});
  assert(safe.moves.length < plain.moves.length, 'nothing was removed');
  console.log(`Проверка в корне: из ${plain.moves.length} ходов осталось ${safe.moves.length}`);
}
