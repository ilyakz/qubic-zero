// Thinks in the background so that the page never freezes.
'use strict';
importScripts('engine.js');

const nets = new Map();

async function loadNet(geo) {
  if (nets.has(geo.name)) return nets.get(geo.name);
  const [manifest, buffer] = await Promise.all([
    fetch('models/' + geo.name + '.json').then(r => { if (!r.ok) throw new Error('no model'); return r.json(); }),
    fetch('models/' + geo.name + '.bin').then(r => { if (!r.ok) throw new Error('no model'); return r.arrayBuffer(); }),
  ]);
  const net = Engine.makeNet(geo, manifest, buffer);
  nets.set(geo.name, net);
  return net;
}

self.onmessage = async ({ data }) => {
  const { id, n, d, k, moves, agent } = data;
  try {
    const geo = Engine.geometry(n, d, k);
    const pos = new Engine.Position(geo, moves);
    let result;
    if (agent.type === 'net' || agent.type === 'pure') {
      const net = agent.type === 'net' ? await loadNet(geo) : new Engine.UniformNet(geo);
      let last = 0;
      const onProgress = (sim, root) => {
        const now = Date.now();
        if (now - last < 150) return;
        last = now;
        self.postMessage({ id, progress: sim / agent.sims, shares: Engine.visitShares(root, geo.cells) });
      };
      const answer = Engine.netMove(pos, net, agent.sims, { vcfBudget: agent.vcfBudget, rootSafety: agent.rootSafety, onProgress });
      result = { cell: answer.cell, value: answer.value, proven: answer.proven, shares: answer.shares };
    } else {
      result = { cell: Engine.bots[agent.type](pos, Math.random) };
    }
    self.postMessage({ id, done: true, result });
  } catch (error) {
    self.postMessage({ id, error: String(error && error.message || error) });
  }
};
