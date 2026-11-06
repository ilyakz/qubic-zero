// Rules, bots, search and the line network in plain JavaScript. Runs in a Web Worker and in Node (for the tests).
'use strict';
(function (exportsRoot) {

  class Geometry {
    constructor(n, d, length) {
      const run = length || n;
      this.n = n; this.d = d; this.k = run;
      this.cells = Math.pow(n, d);
      this.name = Array(d).fill(n).join('x') + (run !== n ? 'k' + run : '');
      this.coords = [];
      for (let c = 0; c < this.cells; c++) {
        const point = []; let rest = c;
        for (let i = 0; i < d; i++) { point.unshift(rest % n); rest = Math.floor(rest / n); }
        this.coords.push(point);
      }
      const steps = [];
      for (let k = 0; k < Math.pow(3, d); k++) {
        const step = []; let rest = k;
        for (let i = 0; i < d; i++) { step.push(rest % 3 - 1); rest = Math.floor(rest / 3); }
        if (step.some(v => v)) steps.push(step);
      }
      const unique = new Map();
      for (const point of this.coords) {
        for (const step of steps) {
          if (!step.every((s, i) => point[i] + s * (run - 1) >= 0 && point[i] + s * (run - 1) < n)) continue;
          const line = [];
          for (let t = 0; t < run; t++) line.push(this.index(point.map((p, i) => p + step[i] * t)));
          line.sort((a, b) => a - b);
          unique.set(line.join(','), line);
        }
      }
      this.lines = [...unique.values()].sort((a, b) => { for (let i = 0; i < run; i++) if (a[i] !== b[i]) return a[i] - b[i]; return 0; });
      this.through = Array.from({ length: this.cells }, () => []);
      this.lines.forEach((line, id) => line.forEach(c => this.through[c].push(id)));
      this.degree = this.through.map(ids => ids.length);
      this.maxDegree = Math.max(...this.degree);
      this.order = [...Array(this.cells).keys()].sort((a, b) => this.degree[b] - this.degree[a] || a - b);
      // Stone hash for the search of threat sequences: 2 × 26 bits fit into a double exactly.
      let seed = 12345;
      const random26 = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return (seed >>> 5) & 0x3ffffff; };
      this.zobrist = [0, 1, 2, 3].map(() => Int32Array.from({ length: this.cells }, random26));
    }
    index(point) { let c = 0; for (const p of point) c = c * this.n + p; return c; }
  }

  const geometries = new Map();
  function geometry(n, d, k) {
    const key = [n, d, k || n].join('x');
    if (!geometries.has(key)) geometries.set(key, new Geometry(n, d, k));
    return geometries.get(key);
  }

  class Position {
    constructor(geo, moves) {
      this.geo = geo;
      this.board = new Int8Array(geo.cells); // 0 — empty, 1 — X, 2 — O
      this.count = [new Int16Array(geo.lines.length), new Int16Array(geo.lines.length)];
      this.side = 0; this.ply = 0; this.h1 = 0; this.h2 = 0; this.won = false;
      if (moves) for (const c of moves) this.play(c);
    }
    play(c) {
      const s = this.side, ids = this.geo.through[c], count = this.count[s];
      this.board[c] = s + 1;
      for (let i = 0; i < ids.length; i++) count[ids[i]]++;
      this.h1 ^= this.geo.zobrist[s * 2][c]; this.h2 ^= this.geo.zobrist[s * 2 + 1][c];
      this.side = 1 - s; this.ply++;
    }
    undo(c) {
      const s = 1 - this.side, ids = this.geo.through[c], count = this.count[s];
      this.board[c] = 0;
      for (let i = 0; i < ids.length; i++) count[ids[i]]--;
      this.h1 ^= this.geo.zobrist[s * 2][c]; this.h2 ^= this.geo.zobrist[s * 2 + 1][c];
      this.side = s; this.ply--;
    }
    get key() { return this.h1 * 67108864 + this.h2; }
    get full() { return this.ply === this.geo.cells; }
    // Cells where this side wins at once.
    wins(side) {
      const mine = this.count[side], theirs = this.count[1 - side], n = this.geo.k, lines = this.geo.lines, out = [];
      for (let l = 0; l < lines.length; l++) {
        if (mine[l] !== n - 1 || theirs[l] !== 0) continue;
        const line = lines[l];
        for (let i = 0; i < n; i++) {
          const c = line[i];
          if (this.board[c] === 0) { if (!out.includes(c)) out.push(c); break; }
        }
      }
      return out;
    }
    completed(cell, side) {
      const ids = this.geo.through[cell];
      for (let i = 0; i < ids.length; i++) if (this.count[side][ids[i]] === this.geo.k) return true;
      return false;
    }
    empties() { const out = []; for (let c = 0; c < this.geo.cells; c++) if (this.board[c] === 0) out.push(c); return out; }
  }

  // State of a game by its moves: who won, which line, the threats of both sides.
  function describe(geo, moves) {
    const pos = new Position(geo);
    let winner = null, line = [];
    for (const c of moves) {
      const side = pos.side;
      pos.play(c);
      if (pos.completed(c, side)) {
        winner = side;
        for (const id of geo.through[c]) if (pos.count[side][id] === geo.k) line = geo.lines[id];
      }
    }
    const finished = winner !== null || moves.length === geo.cells;
    return { winner, line, finished, threats: finished ? [[], []] : [pos.wins(0), pos.wins(1)] };
  }

  // Win by a series of checks: every move makes a threat, the reply is forced. First move or -1.
  function findVcf(pos, budget, depth) {
    const geo = pos.geo, n = geo.k, lines = geo.lines, me = pos.side, op = 1 - me;
    let left = budget === undefined ? 400 : budget;
    const seen = new Set();
    function search(deep) {
      if (--left < 0) return -1;
      const w = pos.wins(me);
      if (w.length) return w[0];
      const threats = pos.wins(op);
      if (threats.length >= 2 || deep === 0) return -1;
      const candidate = new Uint8Array(geo.cells);
      let any = false;
      for (let l = 0; l < lines.length; l++) {
        if (pos.count[me][l] !== n - 2 || pos.count[op][l] !== 0) continue;
        const line = lines[l];
        for (let i = 0; i < n; i++) if (pos.board[line[i]] === 0) { candidate[line[i]] = 1; any = true; }
      }
      if (!any) return -1;
      for (let k = 0; k < geo.cells; k++) {
        const c = geo.order[k];
        if (!candidate[c] || (threats.length && threats[0] !== c)) continue;
        pos.play(c);
        const replies = pos.wins(me);
        if (replies.length >= 2) { pos.undo(c); return c; }
        if (!replies.length) { pos.undo(c); continue; }
        const r = replies[0];
        pos.play(r);
        let found = -1;
        const key = pos.key;
        if (!seen.has(key)) { seen.add(key); found = search(deep - 1); }
        pos.undo(r); pos.undo(c);
        if (found >= 0) return c;
      }
      return -1;
    }
    return search(depth === undefined ? 16 : depth);
  }

  function vcfFor(pos, side, budget) {
    const saved = pos.side;
    pos.side = side;
    const cell = findVcf(pos, budget);
    pos.side = saved;
    return cell;
  }

  // --- bots ---
  const pick = (cells, rng) => cells[Math.floor(rng() * cells.length)];

  const bots = {
    random(pos, rng) { return pick(pos.empties(), rng); },
    osp(pos, rng) {
      const w = pos.wins(pos.side), t = pos.wins(1 - pos.side);
      return pick(w.length ? w : t.length ? t : pos.empties(), rng);
    },
    hsp(pos, rng) {
      const w = pos.wins(pos.side), t = pos.wins(1 - pos.side);
      if (w.length) return pick(w, rng);
      if (t.length) return pick(t, rng);
      let best = -1, cells = [];
      for (const c of pos.empties()) {
        let score = 0;
        for (const l of pos.geo.through[c]) score += pos.count[0][l] + pos.count[1][l];
        if (score > best) { best = score; cells = [c]; } else if (score === best) cells.push(c);
      }
      return pick(cells, rng);
    },
    tsp(pos, rng) {
      const me = pos.side, op = 1 - me;
      const w = pos.wins(me), t = pos.wins(op);
      if (w.length) return pick(w, rng);
      if (t.length) return pick(t, rng);
      const attack = findVcf(pos);
      if (attack >= 0) return attack;
      const scored = pos.empties().map(c => {
        let score = 0;
        for (const l of pos.geo.through[c]) {
          const a = pos.count[me][l], b = pos.count[op][l];
          if (a && b) continue;
          score += b ? 3 * Math.pow(4, b - 1) : Math.pow(4, a);
        }
        return [score + rng(), c];
      }).sort((x, y) => y[0] - x[0]).map(x => x[1]);
      const danger = vcfFor(pos, op);
      const candidates = scored.slice(0, danger >= 0 ? 10 : 3);
      for (const c of candidates) {
        pos.play(c);
        const refuted = findVcf(pos, 150) < 0;
        pos.undo(c);
        if (refuted) return c;
      }
      return danger >= 0 ? danger : scored[0];
    },
  };

  // --- line network: same computation as LineNet in qubic/net.py ---
  class LineNet {
    constructor(geo, manifest, buffer) {
      this.geo = geo;
      this.c = manifest.channels;
      this.blocks = manifest.blocks;
      this.p = {};
      let offset = 0;
      const data = new Float32Array(buffer);
      for (const { name, size } of manifest.params) {
        this.p[name] = data.subarray(offset, offset + size);
        offset += size;
      }
      const cells = geo.cells, L = geo.lines.length, c = this.c;
      this.h = new Float32Array(cells * c);
      this.y = new Float32Array(cells * c);
      this.tmp = new Float32Array(cells * c);
      this.wide = new Float32Array(cells * c * 2);
      this.lineA = new Float32Array(L * c);
      this.lineB = new Float32Array(L * c);
      this.lineScale = 1 / Math.sqrt(geo.k);
      this.cellScale = Float32Array.from(geo.degree, d => 1 / Math.sqrt(d));
      this.degreeNorm = Float32Array.from(geo.degree, d => d / geo.maxDegree);
    }
    // out[r, j] = sum_k in[r, k] * W[j, k] + b[j]
    linear(name, input, output, rows, cIn, cOut) {
      const W = this.p[name + '.weight'], b = this.p[name + '.bias'];
      for (let r = 0; r < rows; r++) {
        const ri = r * cIn, ro = r * cOut;
        for (let j = 0; j < cOut; j++) {
          const wj = j * cIn;
          let sum = b[j];
          for (let k = 0; k < cIn; k++) sum += input[ri + k] * W[wj + k];
          output[ro + j] = sum;
        }
      }
    }
    layerNorm(name, input, output, rows) {
      const c = this.c, gamma = this.p[name + '.weight'], beta = this.p[name + '.bias'];
      for (let r = 0; r < rows; r++) {
        const base = r * c;
        let mean = 0;
        for (let k = 0; k < c; k++) mean += input[base + k];
        mean /= c;
        let variance = 0;
        for (let k = 0; k < c; k++) { const v = input[base + k] - mean; variance += v * v; }
        const scale = 1 / Math.sqrt(variance / c + 1e-5);
        for (let k = 0; k < c; k++) output[base + k] = (input[base + k] - mean) * scale * gamma[k] + beta[k];
      }
    }
    forward(pos) {
      const geo = this.geo, cells = geo.cells, L = geo.lines.length, c = this.c, n = geo.k;
      const h = this.h, y = this.y, tmp = this.tmp, wide = this.wide, la = this.lineA, lb = this.lineB;
      const own = pos.side + 1, opp = 2 - pos.side;
      const W = this.p['embed.weight'], B = this.p['embed.bias'];
      for (let cell = 0; cell < cells; cell++) {
        const v = pos.board[cell];
        const i0 = v === own ? 1 : 0, i1 = v === opp ? 1 : 0, i2 = v === 0 ? 1 : 0, i3 = this.degreeNorm[cell];
        const base = cell * c;
        for (let j = 0; j < c; j++) {
          const w = j * 4;
          h[base + j] = B[j] + i0 * W[w] + i1 * W[w + 1] + i2 * W[w + 2] + i3 * W[w + 3];
        }
      }
      for (let b = 0; b < this.blocks; b++) {
        const prefix = 'blocks.' + b + '.';
        this.layerNorm(prefix + 'norm1', h, tmp, cells);
        this.linear(prefix + 'to_line', tmp, y, cells, c, c);
        la.fill(0);
        for (let l = 0; l < L; l++) {
          const line = geo.lines[l], base = l * c;
          for (let i = 0; i < n; i++) {
            const from = line[i] * c;
            for (let k = 0; k < c; k++) la[base + k] += y[from + k];
          }
          for (let k = 0; k < c; k++) { const v = la[base + k] * this.lineScale; la[base + k] = v > 0 ? v : 0; }
        }
        this.linear(prefix + 'line1', la, lb, L, c, c);
        for (let i = 0; i < L * c; i++) if (lb[i] < 0) lb[i] = 0;
        this.linear(prefix + 'line2', lb, la, L, c, c);
        for (let cell = 0; cell < cells; cell++) {
          const ids = geo.through[cell], base = cell * c, scale = this.cellScale[cell];
          for (let i = 0; i < ids.length; i++) {
            const from = ids[i] * c;
            for (let k = 0; k < c; k++) h[base + k] += la[from + k] * scale;
          }
        }
        this.layerNorm(prefix + 'norm2', h, tmp, cells);
        this.linear(prefix + 'ff1', tmp, wide, cells, c, 2 * c);
        for (let i = 0; i < cells * 2 * c; i++) if (wide[i] < 0) wide[i] = 0;
        this.linear(prefix + 'ff2', wide, y, cells, 2 * c, c);
        for (let i = 0; i < cells * c; i++) h[i] += y[i];
      }
      this.layerNorm('norm', h, tmp, cells);
      const pooled = new Float32Array(2 * c);
      for (let k = 0; k < c; k++) pooled[c + k] = -Infinity;
      for (let cell = 0; cell < cells; cell++) {
        const base = cell * c;
        for (let k = 0; k < c; k++) {
          const v = tmp[base + k];
          pooled[k] += v;
          if (v > pooled[c + k]) pooled[c + k] = v;
        }
      }
      for (let k = 0; k < c; k++) pooled[k] /= cells;
      const hidden = new Float32Array(c);
      this.linear('value1', pooled, hidden, 1, 2 * c, c);
      for (let k = 0; k < c; k++) if (hidden[k] < 0) hidden[k] = 0;
      const out = new Float32Array(1);
      this.linear('value2', hidden, out, 1, c, 1);
      const logits = new Float32Array(cells);
      this.linear('policy', tmp, logits, cells, c, 1);
      return { logits, value: Math.tanh(out[0]) };
    }
    evaluate(pos) {
      const { logits, value } = this.forward(pos);
      const priors = new Float32Array(this.geo.cells);
      let top = -Infinity;
      for (let c = 0; c < logits.length; c++) if (pos.board[c] === 0 && logits[c] > top) top = logits[c];
      let total = 0;
      for (let c = 0; c < logits.length; c++) if (pos.board[c] === 0) { priors[c] = Math.exp(logits[c] - top); total += priors[c]; }
      for (let c = 0; c < logits.length; c++) priors[c] /= total;
      return { priors, value };
    }
  }

  // Network without any knowledge: search and rules only.
  class UniformNet {
    constructor(geo) { this.geo = geo; }
    evaluate(pos) {
      const priors = new Float32Array(this.geo.cells);
      const free = pos.geo.cells - pos.ply;
      for (let c = 0; c < priors.length; c++) if (pos.board[c] === 0) priors[c] = 1 / free;
      return { priors, value: 0 };
    }
  }

  // --- search: the same PUCT as qubic/mcts.py ---
  class Node {
    constructor() { this.analysed = false; this.terminal = null; this.forced = -1; this.moves = null; this.visits = 0; this.qSum = 0; }
  }

  function analyse(node, pos, budget) {
    node.analysed = true;
    const me = pos.side, op = 1 - me;
    const w = pos.wins(me);
    if (w.length) { node.terminal = 1; node.forced = w[0]; return; }
    if (pos.full) { node.terminal = 0; return; }
    const t = pos.wins(op);
    if (t.length >= 2) { node.terminal = -1; node.forced = t[0]; return; }
    if (budget > 0) {
      const cell = findVcf(pos, budget);
      if (cell >= 0) { node.terminal = 1; node.forced = cell; return; }
    }
    if (t.length) {
      node.forced = t[0];
      node.moves = [t[0]]; node.P = [1]; node.N = [0]; node.W = [0]; node.children = [null];
    }
  }

  function expand(node, pos, answer) {
    node.moves = pos.empties();
    const size = node.moves.length;
    node.P = new Float32Array(size);
    let total = 0;
    for (let i = 0; i < size; i++) { node.P[i] = answer.priors[node.moves[i]]; total += node.P[i]; }
    for (let i = 0; i < size; i++) node.P[i] = total > 0 ? node.P[i] / total : 1 / size;
    node.N = new Int32Array(size);
    node.W = new Float32Array(size);
    node.children = new Array(size).fill(null);
    node.qSum = answer.value;
  }

  function select(node, cPuct, fpu) {
    const size = node.moves.length;
    if (size === 1) return 0;
    const sqrt = cPuct * Math.sqrt(node.visits + 1), unvisited = node.qSum / (node.visits + 1) - fpu;
    let best = 0, top = -Infinity;
    for (let i = 0; i < size; i++) {
      const visits = node.N[i];
      const score = (visits > 0 ? node.W[i] / visits : unvisited) + sqrt * node.P[i] / (1 + visits);
      if (score > top) { top = score; best = i; }
    }
    return best;
  }

  // Drops the root moves that give the opponent a win by a series of checks. With more moves than
  // simulations the search itself has no time to check them all. If all checked moves are unsafe, keeps them.
  function removeUnsafe(root, pos, budget, top) {
    if (!root.moves || root.moves.length < 2 || root.forced >= 0) return;
    const order = [...root.moves.keys()].sort((a, b) => root.P[b] - root.P[a]).slice(0, top || 64);
    const unsafe = new Set();
    for (const i of order) {
      pos.play(root.moves[i]);
      if (findVcf(pos, budget) >= 0) unsafe.add(i);
      pos.undo(root.moves[i]);
    }
    if (!unsafe.size || unsafe.size === order.length) return;
    const keep = [...root.moves.keys()].filter(i => !unsafe.has(i));
    let total = 0;
    for (const i of keep) total += root.P[i];
    root.moves = keep.map(i => root.moves[i]);
    root.P = Float32Array.from(keep, i => root.P[i] / total);
    root.N = new Int32Array(keep.length);
    root.W = new Float32Array(keep.length);
    root.children = new Array(keep.length).fill(null);
  }

  // Search from the position; returns the root with the visit counts of its moves.
  function search(pos, net, sims, options) {
    const opts = options || {};
    const cPuct = opts.cPuct || 1.5, fpu = opts.fpu === undefined ? 0.2 : opts.fpu;
    const budget = opts.vcfBudget === undefined ? 400 : opts.vcfBudget;
    const cache = new Map();
    const ask = () => {
      const key = pos.key;
      let answer = cache.get(key);
      if (!answer) { answer = net.evaluate(pos); cache.set(key, answer); }
      return answer;
    };
    const root = new Node();
    analyse(root, pos, 0);
    if (root.moves === null) expand(root, pos, ask());
    if (opts.rootSafety) removeUnsafe(root, pos, opts.rootSafety);
    const stack = [], played = [];
    for (let sim = root.visits; sim < sims; sim++) {
      let node = root;
      stack.length = 0; played.length = 0;
      for (;;) {
        if (!node.analysed) analyse(node, pos, budget);
        if (node.terminal !== null || node.moves === null) break;
        const i = select(node, cPuct, fpu);
        stack.push(node, i);
        const cell = node.moves[i];
        pos.play(cell); played.push(cell);
        if (!node.children[i]) node.children[i] = new Node();
        node = node.children[i];
      }
      let value;
      if (node.terminal !== null) value = node.terminal;
      else { const answer = ask(); expand(node, pos, answer); value = answer.value; }
      for (let k = stack.length - 2; k >= 0; k -= 2) {
        value = -value;
        const parent = stack[k], index = stack[k + 1];
        parent.N[index]++; parent.W[index] += value; parent.visits++; parent.qSum += value;
      }
      for (let k = played.length - 1; k >= 0; k--) pos.undo(played[k]);
      if (opts.onProgress && (sim & 31) === 0) opts.onProgress(sim, root);
    }
    return root;
  }

  function visitShares(root, cells) {
    const shares = new Float32Array(cells);
    let total = 0;
    for (let i = 0; i < root.moves.length; i++) total += root.N[i];
    if (total) for (let i = 0; i < root.moves.length; i++) shares[root.moves[i]] = root.N[i] / total;
    return shares;
  }

  // A move of a network with search: proven sequences and forced moves need no search.
  function netMove(pos, net, sims, options) {
    const budget = options && options.vcfBudget !== undefined ? options.vcfBudget : 400;
    const proven = findVcf(pos, budget);
    if (proven >= 0) return { cell: proven, value: 1, proven: true, shares: null };
    const w = pos.wins(1 - pos.side);
    if (w.length >= 2) return { cell: w[0], value: -1, proven: true, shares: null };
    if (w.length === 1) return { cell: w[0], value: net.evaluate ? net.evaluate(pos).value : 0, proven: false, shares: null };
    const root = search(pos, net, sims, options);
    let best = 0;
    for (let i = 1; i < root.moves.length; i++) if (root.N[i] > root.N[best]) best = i;
    return { cell: root.moves[best], value: root.N[best] ? root.W[best] / root.N[best] : 0, proven: false, shares: visitShares(root, pos.geo.cells), root };
  }

  const api = { Geometry, geometry, Position, describe, findVcf, vcfFor, bots, LineNet, UniformNet, search, netMove, visitShares };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else exportsRoot.Engine = api;
})(typeof self !== 'undefined' ? self : this);
