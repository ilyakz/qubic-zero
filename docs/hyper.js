// Views of the 4D board on a canvas: four stacks of layers, or a real projection of a tesseract.
'use strict';

function rotatePlane(m, i, j, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), a = m[i], b = m[j];
  for (let k = 0; k < a.length; k++) {
    const p = a[k], q = b[k];
    a[k] = p * c - q * s;
    b[k] = p * s + q * c;
  }
}

// Без этого ошибки округления за долгое вращение понемногу искажали бы фигуру
function orthonormalize(m) {
  m.forEach((row, i) => {
    for (let j = 0; j < i; j++) {
      const dot = row.reduce((sum, v, k) => sum + v * m[j][k], 0);
      row.forEach((v, k) => { row[k] = v - dot * m[j][k]; });
    }
    const length = Math.hypot(...row);
    row.forEach((v, k) => { row[k] = v / length; });
  });
}

function identity(n) {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

class HyperView {
  constructor(canvas, geo, options) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.geo = geo;
    this.mode = options.mode;
    this.onPlay = options.onPlay;
    this.labels = options.labels;
    this.onCaption = options.onCaption;
    this.onSpin = options.onSpin;
    this.auto = options.auto !== false;
    this.yaw = -0.66; this.pitch = 1.15;
    this.m3 = identity(3); this.m4 = identity(4);
    rotatePlane(this.m3, 0, 2, 0.62); rotatePlane(this.m3, 1, 2, 0.42);
    rotatePlane(this.m4, 0, 3, 0.55); rotatePlane(this.m4, 1, 3, 0.34);
    this.spin = 0; this.turns = 0; this.fling = null; this.velocity = null; this.held = false;
    this.hover = -1; this.inside = false; this.drag = null; this.alive = true;
    this.lineStart = 0; this.demoIndex = -1;
    const half = (geo.n - 1) / 2;
    this.half = half;
    this.unit = geo.coords.map(p => p.map(v => (v - half) / half));
    this.demos = this.pickDemos();
    this.points = new Array(geo.cells);
    this.tiles = new Array(geo.cells);
    this.state = { moves: [], line: [], winner: null, threats: [[], []], shares: null, canMove: false, showPeers: true, showHints: true };
    this.readColors();
    this.resize();
    this.onResize = () => { this.resize(); this.request(); };
    window.addEventListener('resize', this.onResize);
    this.bind();
    this.loop();
  }

  destroy() {
    this.alive = false;
    window.removeEventListener('resize', this.onResize);
  }

  readColors() {
    const style = getComputedStyle(document.documentElement), get = name => style.getPropertyValue(name).trim();
    this.colors = { x: get('--x'), o: get('--o'), xSoft: get('--x-soft'), oSoft: get('--o-soft'), cell: get('--cell'), ink: get('--ink'), ink2: get('--ink2'),
      peer: get('--peer'), heat: get('--heat'), frame: get('--border'), accent: get('--accent') };
  }

  // По одной линии каждого вида: они и объясняют, что такое линия в четырёх измерениях
  pickDemos() {
    const { coords, lines } = this.geo;
    const kinds = ['0001', '1000', '1001', '1111'];
    return kinds.map(kind => {
      const found = lines.filter(line => coords[line[0]].map((v, i) => (v !== coords[line[1]][i] ? 1 : 0)).join('') === kind);
      return found[Math.floor(found.length * 0.62)];
    });
  }

  resize() {
    const width = this.canvas.parentElement.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    let height;
    if (this.mode === 'cubes') {
      // В один ряд, пока клетки различимы: тогда прямая линия и на экране прямая
      this.cols = width >= 600 ? this.geo.n : 2;
      const rows = Math.ceil(this.geo.n / this.cols);
      this.cellWidth = width / this.cols;
      this.rowHeight = Math.round(Math.min(this.cellWidth * 1.5, 380));
      height = rows * this.rowHeight;
    } else {
      height = Math.round(Math.min(width * 0.86, 620));
    }
    this.width = width; this.height = height;
    this.canvas.style.height = height + 'px';
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- стопки слоёв: клетка лежит на плоскости, слои разведены по высоте ---
  placeTile(w, u, v, h) {
    const cube = this.cube[w];
    const U = u * this.cosY - v * this.sinY, V = u * this.sinY + v * this.cosY;
    const Y = V * this.cosT - h * this.sinT, depth = V * this.sinT + h * this.cosT;
    const k = 18 / (18 - depth);
    return { x: cube.x + cube.size * U * k, y: cube.y + cube.size * Y * k, depth, k };
  }

  layoutTiles() {
    const { n, cells, coords } = this.geo;
    this.cosY = Math.cos(this.yaw); this.sinY = Math.sin(this.yaw);
    this.cosT = Math.cos(this.pitch); this.sinT = Math.sin(this.pitch);
    this.spread = 2.2;
    const size = Math.min((this.cellWidth - 6) / 6.2, (this.rowHeight - 34) / 8.5);
    this.cube = [];
    for (let w = 0; w < n; w++) {
      const col = w % this.cols, row = Math.floor(w / this.cols);
      this.cube.push({ x: this.cellWidth * (col + 0.5), y: this.rowHeight * row + (this.rowHeight + 24) / 2, size, top: this.rowHeight * row + 16 });
    }
    const half = this.half, gap = 0.45;
    for (let c = 0; c < cells; c++) {
      const [w, z, y, x] = coords[c];
      const u = x - half, v = y - half, h = (half - z) * this.spread;
      const center = this.placeTile(w, u, v, h);
      center.u = u; center.v = v; center.h = h; center.w = w;
      center.corners = [[-gap, -gap], [gap, -gap], [gap, gap], [-gap, gap]].map(([a, b]) => this.placeTile(w, u + a, v + b, h));
      this.points[c] = center;
    }
  }

  // Каркас куба связывает четыре слоя в один предмет
  drawCubeFrames() {
    const ctx = this.ctx, { n } = this.geo, box = n / 2, tall = this.half * this.spread + 0.2;
    ctx.lineWidth = 1;
    ctx.strokeStyle = this.colors.frame;
    ctx.textAlign = 'center';
    ctx.font = '600 13px system-ui, sans-serif';
    for (let w = 0; w < n; w++) {
      const corner = (a, b, h) => this.placeTile(w, a * box, b * box, h * tall);
      const signs = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
      ctx.beginPath();
      signs.forEach(([a, b], i) => {
        const [c, d] = signs[(i + 1) % 4];
        for (const h of [-1, 1]) {
          const s = corner(a, b, h), e = corner(c, d, h);
          ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y);
        }
        const s = corner(a, b, -1), e = corner(a, b, 1);
        ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y);
      });
      ctx.stroke();
      ctx.fillStyle = this.colors.ink2;
      ctx.fillText(`${this.labels.cube} ${w + 1}`, this.cube[w].x, this.cube[w].top);
    }
  }

  tilePath(c, grow) {
    const ctx = this.ctx, p = this.points[c], k = grow || 1;
    ctx.beginPath();
    p.corners.forEach((q, i) => {
      const x = p.x + (q.x - p.x) * k, y = p.y + (q.y - p.y) * k;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath();
  }

  drawMarkOnTile(c, side) {
    const ctx = this.ctx, p = this.points[c], r = 0.25;
    const at = (a, b) => this.placeTile(p.w, p.u + a, p.v + b, p.h);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = Math.max(1.6, this.cube[0].size * 0.075 * p.k);
    ctx.lineCap = 'round';
    ctx.beginPath();
    if (side === 0) {
      for (const [a, b, c2, d] of [[-r, -r, r, r], [r, -r, -r, r]]) {
        const s = at(a, b), e = at(c2, d);
        ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y);
      }
    } else {
      for (let i = 0; i <= 24; i++) {
        const angle = i / 24 * Math.PI * 2, q = at(Math.cos(angle) * r, Math.sin(angle) * r);
        if (i) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y);
      }
    }
    ctx.stroke();
  }

  // --- тессеракт: поворот в 4D, перспектива в 3D, поворот в 3D, перспектива на экран ---
  from4d(w, z, y, x) {
    const a = this.m4, b = this.m3;
    const out = i => a[i][0] * x + a[i][1] * y + a[i][2] * z + a[i][3] * w;
    const near = 3 / (3 - out(3));
    const X = out(0) * near, Y = out(1) * near, Z = out(2) * near;
    const screen = i => b[i][0] * X + b[i][1] * Y + b[i][2] * Z;
    const depth = screen(2), k = 9 / (9 + depth);
    const size = (Math.min(this.width, this.height) / 2 - 14) / 2.8;
    return { x: this.width / 2 + screen(0) * size * k, y: this.height / 2 + screen(1) * size * k, depth, k: k * near };
  }

  // Повороты копятся в матрицах, а не в углах: нет ни упоров, ни рывков, крутить можно сколько угодно
  turn(kind, dx, dy) {
    if (this.mode === 'cubes') {
      this.yaw -= dx * 0.008;
      this.pitch = Math.max(0.6, Math.min(1.4, this.pitch - dy * 0.006));
    } else {
      const m = kind === '4d' ? this.m4 : this.m3, axis = kind === '4d' ? 3 : 2;
      rotatePlane(m, 0, axis, dx * 0.008);
      rotatePlane(m, 1, axis, dy * 0.008);
      if (kind === '4d') this.moveSpin(dx);
      this.tidy();
    }
    this.dirty = true;
  }

  moveSpin(amount) {
    this.spin += amount;
    if (this.onSpin) this.onSpin(this.spin);
  }

  tidy() {
    if (++this.turns % 500 === 0) { orthonormalize(this.m3); orthonormalize(this.m4); }
  }

  press() {
    this.fling = null; this.velocity = null; this.held = true;
    this.lastTurn = performance.now();
  }

  moveBy(kind, dx, dy) {
    this.turn(kind, dx, dy);
    const now = performance.now(), perFrame = 16.7 / Math.max(1, now - this.lastTurn), old = this.velocity;
    this.lastTurn = now;
    const smooth = (v, was) => Math.max(-40, Math.min(40, v * perFrame * 0.6 + was * 0.4));
    this.velocity = { kind, dx: smooth(dx, old ? old.dx : dx * perFrame), dy: smooth(dy, old ? old.dy : dy * perFrame) };
  }

  // Отпустили на ходу: фигура докручивается сама и плавно останавливается
  letGo() {
    const moving = this.velocity && performance.now() - this.lastTurn < 90;
    if (moving && !matchMedia('(prefers-reduced-motion: reduce)').matches) this.fling = this.velocity;
    this.velocity = null; this.held = false;
  }

  layoutDots() {
    for (let c = 0; c < this.geo.cells; c++) this.points[c] = this.from4d(...this.unit[c]);
  }

  drawTesseractFrame() {
    const ctx = this.ctx, corners = [];
    for (let i = 0; i < 16; i++) corners.push([i & 8 ? 1 : -1, i & 4 ? 1 : -1, i & 2 ? 1 : -1, i & 1 ? 1 : -1]);
    const projected = corners.map(p => this.from4d(...p));
    ctx.lineWidth = 1;
    ctx.strokeStyle = this.colors.frame;
    ctx.beginPath();
    for (let i = 0; i < 16; i++) for (const bit of [1, 2, 4, 8]) {
      if (i & bit) continue;
      const s = projected[i], e = projected[i | bit];
      ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y);
    }
    ctx.stroke();
  }

  drawDot(c, side, share, top, threat, dim) {
    const ctx = this.ctx, p = this.points[c], c0 = this.colors;
    const marked = side !== undefined;
    const r = marked ? Math.max(5, 6.5 * p.k) : Math.max(2, 2.6 * p.k);
    ctx.globalAlpha = dim ? 0.2 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    if (marked) {
      ctx.fillStyle = side === 0 ? c0.x : c0.o;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = Math.max(1.3, r * 0.26);
      ctx.lineCap = 'round';
      ctx.beginPath();
      if (side === 0) {
        const a = r * 0.45;
        ctx.moveTo(p.x - a, p.y - a); ctx.lineTo(p.x + a, p.y + a); ctx.moveTo(p.x + a, p.y - a); ctx.lineTo(p.x - a, p.y + a);
      } else {
        ctx.arc(p.x, p.y, r * 0.5, 0, Math.PI * 2);
      }
      ctx.stroke();
    } else {
      if (share > 0.004) {
        ctx.fillStyle = `rgba(${c0.heat},${Math.min(1, 0.35 + 0.65 * share / Math.max(top, 0.05)).toFixed(2)})`;
      } else {
        ctx.fillStyle = c0.ink2;
        ctx.globalAlpha = dim ? 0.1 : Math.min(0.6, 0.2 + 0.28 * p.k);
      }
      ctx.fill();
      if (threat !== null) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2;
        ctx.strokeStyle = threat === 0 ? c0.x : c0.o;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    return r;
  }

  drawTile(c, side, share, top, threats, dim) {
    const ctx = this.ctx, c0 = this.colors;
    ctx.globalAlpha = dim ? 0.22 : 1;
    this.tilePath(c);
    ctx.fillStyle = side === undefined ? c0.cell : side === 0 ? c0.x : c0.o;
    ctx.fill();
    if (side === undefined) {
      if (share > 0.004) {
        ctx.fillStyle = `rgba(${c0.heat},${Math.min(0.95, 0.18 + 0.77 * share / Math.max(top, 0.05)).toFixed(2)})`;
        ctx.fill();
      }
      if (threats[0]) { ctx.fillStyle = c0.xSoft; ctx.fill(); }
      if (threats[1]) { ctx.fillStyle = c0.oSoft; ctx.fill(); }
    } else {
      this.drawMarkOnTile(c, side);
    }
    ctx.globalAlpha = 1;
  }

  polyline(line, part) {
    const ctx = this.ctx;
    const last = (line.length - 1) * (part === undefined ? 1 : part), whole = Math.floor(last);
    ctx.beginPath();
    for (let i = 0; i <= whole; i++) {
      const p = this.points[line[i]];
      if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
    }
    if (whole < line.length - 1 && last > whole) {
      const a = this.points[line[whole]], b = this.points[line[whole + 1]], f = last - whole;
      ctx.lineTo(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f);
    }
    ctx.stroke();
  }

  beam(line, color, part) {
    const ctx = this.ctx;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = 20;
    ctx.strokeStyle = color; ctx.lineWidth = 7;
    this.polyline(line, part);
    ctx.restore();
    ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 2;
    this.polyline(line, part);
  }

  outline(c, color, width) {
    const ctx = this.ctx;
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.lineJoin = 'round';
    if (this.mode === 'cubes') {
      this.tilePath(c, 1.02);
    } else {
      const p = this.points[c];
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(6, 6.5 * p.k) + 2, 0, Math.PI * 2);
    }
    ctx.stroke();
  }

  demoLine() {
    if (this.state.moves.length || this.hover >= 0 || this.held) return null;
    return this.demos[this.demoIndex] || null;
  }

  draw() {
    const ctx = this.ctx, st = this.state, geo = this.geo, c0 = this.colors, cubes = this.mode === 'cubes';
    ctx.clearRect(0, 0, this.width, this.height);
    if (cubes) { this.layoutTiles(); this.drawCubeFrames(); } else { this.layoutDots(); this.drawTesseractFrame(); }

    const owner = new Map(st.moves.map((c, i) => [c, i % 2]));
    const last = st.moves[st.moves.length - 1];
    const finished = st.line.length > 0, won = new Set(st.line);
    const demo = this.demoLine(), inDemo = new Set(demo || []);
    const peers = new Set(), through = [];
    if (this.hover >= 0 && st.showPeers && !finished) {
      for (const id of geo.through[this.hover]) { through.push(geo.lines[id]); for (const c of geo.lines[id]) peers.add(c); }
    }
    let top = 0;
    if (st.shares) for (let c = 0; c < geo.cells; c++) top = Math.max(top, st.shares[c]);
    const threat = st.showHints ? [new Set(st.threats[0]), new Set(st.threats[1])] : [new Set(), new Set()];
    const order = [...Array(geo.cells).keys()];
    if (cubes) order.sort((a, b) => this.points[a].depth - this.points[b].depth);
    else order.sort((a, b) => this.points[b].depth - this.points[a].depth);
    this.order = order;

    const paint = c => {
      const side = owner.get(c), share = st.shares ? st.shares[c] : 0, dim = finished && !won.has(c);
      if (cubes) this.drawTile(c, side, share, top, [threat[0].has(c), threat[1].has(c)], dim);
      else this.drawDot(c, side, share, top, threat[0].has(c) ? 0 : threat[1].has(c) ? 1 : null, dim);
    };
    for (const c of order) if (!won.has(c)) paint(c);

    if (through.length) {
      ctx.globalAlpha = cubes ? 0.5 : 0.75;
      ctx.lineWidth = 1.4;
      ctx.lineCap = 'round';
      ctx.strokeStyle = c0.peer;
      for (const line of through) this.polyline(line);
      ctx.globalAlpha = 1;
      for (const c of peers) if (c !== this.hover) this.outline(c, c0.peer, 1.6);
    }
    if (demo) {
      this.beam(demo, c0.accent);
      for (const c of inDemo) this.outline(c, c0.ink, 2);
    }
    if (this.hover >= 0) this.outline(this.hover, c0.ink, 2.4);
    if (last !== undefined && !finished) this.outline(last, c0.ink, 1.5);
    if (finished) {
      const part = Math.min(1, (performance.now() - this.lineStart) / 700);
      this.beam(st.line, st.winner === 0 ? c0.x : c0.o, part);
      for (const c of st.line) paint(c);
      if (part < 1) this.dirty = true;
    }
  }

  pick(mx, my) {
    if (!this.order) return -1;
    if (this.mode === 'cubes') {
      for (let i = this.order.length - 1; i >= 0; i--) {
        const c = this.order[i], q = this.points[c].corners;
        let inside = true, sign = 0;
        for (let k = 0; k < 4 && inside; k++) {
          const a = q[k], b = q[(k + 1) % 4], cross = (b.x - a.x) * (my - a.y) - (b.y - a.y) * (mx - a.x);
          if (cross !== 0) { if (sign && Math.sign(cross) !== sign) inside = false; sign = Math.sign(cross); }
        }
        if (inside) return c;
      }
      return -1;
    }
    let best = -1, bestDistance = 196;
    for (let c = 0; c < this.geo.cells; c++) {
      const p = this.points[c], dx = p.x - mx, dy = p.y - my, d = dx * dx + dy * dy;
      if (d < bestDistance) { best = c; bestDistance = d; }
    }
    return best;
  }

  bind() {
    const canvas = this.canvas;
    const local = e => { const box = canvas.getBoundingClientRect(); return [e.clientX - box.left, e.clientY - box.top]; };
    canvas.addEventListener('pointerdown', e => {
      const [x, y] = local(e);
      const spinning = this.fling && Math.abs(this.fling.dx) + Math.abs(this.fling.dy) > 1;
      this.press();
      this.drag = { x, y, lastX: x, lastY: y, moved: false, kind: e.shiftKey ? '4d' : '3d', onlyStop: spinning };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', e => {
      const [x, y] = local(e);
      this.inside = true;
      const drag = this.drag;
      if (drag) {
        if (Math.abs(x - drag.x) + Math.abs(y - drag.y) > 5) drag.moved = true;
        if (drag.moved) {
          this.moveBy(drag.kind, x - drag.lastX, y - drag.lastY);
          drag.lastX = x; drag.lastY = y;
        }
        return;
      }
      const cell = this.pick(x, y);
      if (cell !== this.hover) { this.hover = cell; this.request(); this.caption(); }
      canvas.style.cursor = cell >= 0 && this.state.canMove && !this.state.moves.includes(cell) ? 'pointer' : 'grab';
    });
    canvas.addEventListener('pointerup', e => {
      const drag = this.drag;
      this.drag = null;
      this.letGo();
      if (!drag || drag.moved || drag.onlyStop) return;
      const cell = this.pick(...local(e));
      if (cell >= 0 && this.state.canMove && !this.state.moves.includes(cell)) this.onPlay(cell);
    });
    canvas.addEventListener('pointercancel', () => { this.drag = null; this.letGo(); });
    canvas.addEventListener('pointerleave', () => {
      this.inside = false;
      if (this.hover !== -1) { this.hover = -1; this.request(); this.caption(); }
    });
  }

  caption() {
    if (!this.onCaption) return;
    let text = '';
    if (this.hover >= 0) {
      const [w, z, y, x] = this.geo.coords[this.hover];
      text = this.labels.cell(w + 1, z + 1, y + 1, x + 1, this.geo.through[this.hover].length);
    } else if (!this.state.moves.length && this.demoIndex >= 0) {
      text = this.labels.demos[this.demoIndex];
    }
    if (text !== this.lastCaption) { this.lastCaption = text; this.onCaption(text); }
  }

  update(state) {
    const had = this.state.line.length > 0;
    Object.assign(this.state, state);
    if (!had && this.state.line.length) this.lineStart = performance.now();
    this.caption();
    this.request();
  }

  request() { this.dirty = true; }

  loop() {
    if (!this.alive) return;
    const finished = this.state.line.length > 0;
    const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!this.state.moves.length) {
      const index = calm ? 0 : Math.floor(performance.now() / 2800) % this.demos.length;
      if (index !== this.demoIndex) { this.demoIndex = index; this.caption(); this.dirty = true; }
    }
    if (this.fling) {
      const f = this.fling;
      this.turn(f.kind, f.dx, f.dy);
      f.dx *= 0.98; f.dy *= 0.98;
      if (Math.abs(f.dx) + Math.abs(f.dy) < 0.15) this.fling = null;
    }
    // Само вращается всегда, кроме момента, когда игрок целится в клетку
    const aiming = this.inside && this.state.canMove;
    if (!calm && !this.held && !this.fling && !aiming) {
      if (this.mode === 'tesseract' && this.auto) {
        rotatePlane(this.m4, 0, 3, 0.005);
        rotatePlane(this.m4, 1, 3, 0.0031);
        rotatePlane(this.m4, 2, 3, 0.0017);
        this.moveSpin(0.6);
        this.tidy();
        this.dirty = true;
      }
      if (finished) this.turn('3d', 0.375, 0);
    }
    if (this.dirty) { this.dirty = false; this.draw(); }
    requestAnimationFrame(() => this.loop());
  }
}
if (typeof self !== 'undefined') self.HyperView = HyperView;
