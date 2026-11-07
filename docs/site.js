// The part of the page below the game: how the network learned and what it achieved.
'use strict';
let summary = null;

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

// One axis, thin lines, a legend above and a tooltip on hover.
function lineChart(el, series, height) {
  const shown = series.filter(s => s.points.length);
  if (!shown.length) { el.innerHTML = ''; return; }
  const W = el.clientWidth || 320, H = height, m = { l: 38, r: 44, t: 8, b: 26 }, lo = 0, hi = 100;
  const xs = shown.flatMap(s => s.points.map(p => p.x));
  const xMin = Math.min(...xs), xMax = Math.max(Math.max(...xs), xMin + 1);
  const X = x => m.l + (x - xMin) / (xMax - xMin) * (W - m.l - m.r);
  const Y = y => m.t + (hi - y) / (hi - lo) * (H - m.t - m.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${t.chartAria}">`;
  for (const v of niceTicks(lo, hi, 4)) {
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>`;
    svg += `<text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11.5" fill="var(--muted)">${v}</text>`;
  }
  svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  for (const v of niceTicks(xMin, xMax, Math.max(2, Math.floor(W / 90))).filter(Number.isInteger)) svg += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle" font-size="11.5" fill="var(--muted)">${v}</text>`;
  for (const s of shown) {
    svg += `<path d="${s.points.map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ' ' + Y(p.y).toFixed(1)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const e = s.points[s.points.length - 1];
    svg += `<circle cx="${X(e.x)}" cy="${Y(e.y)}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
  }
  svg += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" visibility="hidden"/><g class="dots"></g>`;
  svg += `<rect x="${m.l}" y="${m.t}" width="${Math.max(1, W - m.l - m.r)}" height="${H - m.t - m.b}" fill="transparent" class="hit"/></svg>`;
  el.innerHTML = `<div class="chart-legend">${shown.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')}</div>${svg}<div class="tip" hidden></div>`;
  const hit = el.querySelector('.hit'), tip = el.querySelector('.tip'), cross = el.querySelector('.cross'), dots = el.querySelector('.dots'), box = el.querySelector('svg');
  const all = [...new Set(xs)].sort((a, b) => a - b);
  hit.addEventListener('mousemove', ev => {
    const r = box.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const x = all.reduce((b, c) => Math.abs(X(c) - px) < Math.abs(X(b) - px) ? c : b, all[0]);
    cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x)); cross.setAttribute('visibility', 'visible');
    let rows = `<div class="tip-head">${t.iteration} ${x}</div>`, marks = '';
    for (const s of shown) {
      const p = s.points.find(q => q.x === x);
      if (!p) continue;
      marks += `<circle cx="${X(x)}" cy="${Y(p.y)}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
      rows += `<div><i style="background:${s.color}"></i>${s.name}: <b>${Math.round(p.y)}%</b></div>`;
    }
    dots.innerHTML = marks; tip.innerHTML = rows; tip.hidden = false;
    tip.style.left = Math.min(Math.max(0, X(x) * r.width / W + 12), el.clientWidth - tip.offsetWidth) + 'px';
    tip.style.top = (box.offsetTop + 6) + 'px';
  });
  hit.addEventListener('mouseleave', () => { tip.hidden = true; dots.innerHTML = ''; cross.setAttribute('visibility', 'hidden'); });
}

const RACE_COLORS = { n44: 'var(--control)', c44: 'var(--c2)', c44n: 'var(--c4)', c44s: 'var(--c3)', x_b48: 'var(--c1)', x_gl: 'var(--accent)' };

function drawRace() {
  if (!summary) return;
  const runs = summary.race.filter(r => r.points.length);
  const fresh = runs.filter(r => r.run !== 'n44');
  const limit = fresh.length ? Math.max(...fresh.map(r => r.points[r.points.length - 1][0])) : Infinity;
  const pick = i => runs.map(r => ({ name: t.raceNames[r.run] || r.run, color: RACE_COLORS[r.run] || 'var(--accent)',
    points: r.points.filter(p => p[i] != null && p[0] <= limit).map(p => ({ x: p[0], y: p[i] })) }));
  lineChart($('race-all'), pick(1), 250);
  lineChart($('race-second'), pick(2), 250);
}

function boardName(b) {
  const size = Array(b.d).fill(b.n).join('×');
  if (b.n === 15) return `${t.gomoku} ${size}`;
  return b.k !== b.n ? `${size}, ${t.inRow(b.k)}` : size;
}

function resultText(b) {
  if (b.exact && b.exact.mistakes != null) return t.exactResult(b.exact.mistakes, (b.exact.decisions || 0) + (b.exact.positions || 0));
  if (b.exact) return t.exactResult(b.exact.blunders, b.exact.decisions);
  if (!b.vs) return t.inTraining;
  const pct = o => b.vs[o] ? Math.round(b.vs[o].score) + '%' : '—';
  return `${t.vsTsp} ${pct('tsp')}, ${t.vsPure} ${pct('pure')}`;
}

function drawResults() {
  if (!summary) return;
  const value = v => v === 0 ? t.valueDraw : v === 1 ? t.valueFirst : '—';
  const rows = summary.boards.map(b => `<tr${b.d === 4 ? ' class="frontier"' : ''}>
    <td class="name"><a href="?board=${b.name}">${boardName(b)}</a></td><td>${b.cells}</td><td>${b.lines}</td><td>${String(b.per_cell).replace('.', t.decimal)}</td>
    <td class="name">${value(b.value)}</td><td class="name">${b.arch ? `${t.nets[b.arch] || b.arch} ${b.channels}×${b.blocks}` : '—'}</td><td class="name">${resultText(b)}</td></tr>`).join('');
  $('results-table').innerHTML = `<tr><th class="name">${t.colBoard}</th><th>${t.colCells}</th><th>${t.colLines}</th><th>${t.colPerCell}</th><th class="name">${t.colValue}</th><th class="name">${t.colNet}</th><th class="name">${t.colResult}</th></tr>${rows}`;
}

function drawSite() {
  drawRace();
  drawResults();
}

fetch('data/summary.json').then(r => r.json()).then(data => { summary = data; drawSite(); }).catch(() => { $('about').hidden = true; });
let siteResize = null;
window.addEventListener('resize', () => { clearTimeout(siteResize); siteResize = setTimeout(drawRace, 150); });
