'use strict';
const $ = id => document.getElementById(id);

const TEXT = {
  ru: {
    title: 'Крестики-нолики в четырёх измерениях',
    brand: 'Qubic Zero', chartAria: 'График силы сетей по итерациям обучения', iteration: 'итерация', decimal: ',',
    aboutRulesTitle: 'Что такое четвёртое измерение', ruleCubes: 'Четыре куба', ruleCubesText: 'Доска это четыре куба 4×4×4. Номер куба и есть четвёртая координата, а проекция выше показывает их один внутри другого.',
    ruleLine: 'Линия из четырёх', ruleLineText: 'Как и в обычных крестиках-ноликах, надо занять клетки одной линии. Она лежит внутри одного куба или идёт через все четыре, по одной клетке в каждом.',
    ruleThreat: 'Угрозы отовсюду', ruleThreatText: 'Линий 520, и через клетку их проходит до 15. Партия обычно решается примерно за 14 ходов, хотя клеток 256.',
    aboutLearnTitle: 'Как сеть научилась играть', aboutLearnText: 'Никто не показывал ей партии людей. Сеть играет сама с собой и с ботами, учится на этих партиях и снова играет. Поиск по дереву считает варианты, а сеть подсказывает ему, какие ходы смотреть и кто выигрывает.',
    raceAll: 'Счёт против бота tsp, %', raceAllNote: 'tsp ищет форсированные выигрыши и не допускает чужих. Половину партий сеть ходит первой.', raceSecond: 'Против tsp вторым ходом, %', raceSecondNote: 'Самое трудное: на этой доске у первого игрока большое преимущество.',
    nets: { linenet: 'сеть на линиях', linenet2: 'новая сеть на линиях', conv4d: '4D-свёртки', conv4d_lines: '4D-свёртки и подсказки о линиях', conv2d: 'свёртки', resnet: '3D-свёртки', lines: '3D-свёртки и линии', hybrid: 'гибрид свёрток и линий' },
    raceNames: { n44: 'сеть на линиях, первая', c44: '4D-свёртки', c44n: 'свёртки, подсказки о линиях, новый шум', c44s: 'маленькая, до Gumbel', x_b48: 'побольше, Gumbel с нуля', x_gl: 'маленькая, Gumbel и лига, в этой игре' },
    aboutResultsTitle: 'Результаты на всех досках', aboutResultsText: 'У каждой доски своя сеть. На маленьких досках решатель знает цену каждой позиции, поэтому там результат точный.',
    colBoard: 'Доска', colCells: 'Клеток', colLines: 'Линий', colPerCell: 'Линий через клетку', colValue: 'Идеальная игра', colNet: 'Сеть', colResult: 'Результат',
    valueDraw: 'ничья', valueFirst: 'выигрывает первый', vsTsp: 'tsp', vsPure: 'поиск без сети', inTraining: 'сеть ещё обучается', exactResult: (m, n) => `${m} ошибок на ${n.toLocaleString('ru')} решений, проверено перебором`,
    aboutFindingsTitle: 'Что показали эксперименты',
    find1: 'Одна строка на 7 лет', find1Text: 'В первой версии 2019 года сеть оценки училась не на исходе партии, а на том, чей ход. Ошибка обучения при этом падала почти до нуля и выглядела отлично.',
    find2: 'Больше не значит сложнее', find2Text: '5×5×5 вдвое больше 4×4×4, но заканчивается вничью. Похоже, сложность зависит от числа линий через клетку, а не от размера доски.',
    find3: 'Самая длинная линия самая скучная', find3Text: 'На 5×5 с пятью в ряд второй игрок всегда сводит вничью: каждой линии хватает своей пары клеток. Это доказательство, а не наблюдение.',
    find4: 'Универсальной сети нет', find4Text: 'На 4×4×4×4 лучше всех сыграли свёртки с подсказками о линиях, а на 9×9 победила простая сеть на линиях, втрое с лишним меньше свёрточной. Для каждой доски сеть выбирали прямые матчи.',
    readMore: 'Подробнее об обучении и находках в README',
    lead: 'Займите четыре клетки одной линии на доске 4×4×4×4: 256 клеток и 520 линий. Соперник это нейросеть, которая училась сама, без партий людей. Всё считается прямо в браузере, а если 4D это слишком, есть доски попроще, от 3×3.',
    board: 'Доска', crosses: '× крестики, ходят первыми', noughts: '○ нолики', strength: 'Сила сети', view: 'Вид',
    hints: 'Подсвечивать угрозы', thoughts: 'Показывать мысли сети', peers: 'Показывать линии через клетку',
    newGame: 'Новая партия', undo: 'Отменить ход', evaluation: 'оценка сети', spread: 'Расстояние между слоями',
    legendX: 'крестики', legendO: 'нолики', legendXT: 'здесь × замыкает линию', legendOT: 'здесь ○ замыкает линию',
    howTitle: 'Как играть',
    how0: 'На плоских досках линия идёт по строке, столбцу или диагонали. Обычно нужно занять её целиком, а на больших полях хватает части: в гомоку на поле 15×15 нужно поставить пять в ряд, шесть и больше тоже считается.',
    goal: k => `Соберите ${k} в ряд`, gomoku: 'Гомоку', inRow: k => `${k} в ряд`, need: k => `Нужно ${k} в ряд.`,
    lineFlat: kind => `Линия на доске: ${['прямая по строке или столбцу', 'диагональ'][kind]}.`,
    how1: 'Чтобы выиграть, займите все клетки одной линии. Клетка с тем же местом в соседнем слое лежит прямо над ней или под ней, поэтому линия может идти через все слои.',
    how2: 'Самое сильное оружие: создать сразу две угрозы, закрыть обе соперник не успеет. Углы и центры лежат на большем числе линий и потому ценнее.',
    how3: 'На доске 4×4×4×4 четвёртое измерение — это номер куба. Клетки на одном месте в разных кубах лежат на одной линии, а линия может идти и наискосок через все четыре куба. Наведите на клетку, чтобы увидеть все линии через неё. Вид «Гиперкуб» показывает то же самое как настоящую проекцию тессеракта: куб внутри куба. Перетаскивайте мышью, чтобы вращать.',
    footer: 'Исходный код и история проекта:',
    you: 'Вы', net: 'Нейросеть', pure: 'Поиск без сети', tsp: 'Бот tsp: ищет серии шахов', hsp: 'Бот hsp: эвристика', osp: 'Бот osp: простой', random: 'Случайные ходы',
    fast: 'Быстро', medium: 'Средне', strong: 'Сильно', layers: 'Слои', volume: 'Объём', board4d: 'Таблица досок', cubes: 'Четыре куба', tesseract: 'Гиперкуб', cube: 'Куб', rotate4d: 'Вращение в 4D', auto4d: 'само', note4d: 'Четвёртое измерение — это номер куба. Линия лежит внутри одного куба или идёт через все четыре: по одной клетке в каждом.', note4dHyper: 'Это настоящая проекция тессеракта: каждая точка это клетка, а один куб лежит внутри другого. Тяните мышью, чтобы вращать в 3D, а с Shift или за полоску ниже — в четвёртом измерении. Можно раскрутить и отпустить. Пока вы целитесь в клетку, сам он не вращается.', demo4d: ['Пример: прямая внутри одного куба', 'Пример: одна и та же клетка в каждом кубе', 'Пример: наискосок через все кубы', 'Пример: главная диагональ, меняются все четыре координаты'], cell4d: (w, z, y, x, count) => `Куб ${w}, слой ${z}, ряд ${y}, клетка ${x}. Линий через неё: ${count}`,
    layer: 'Слой', noModel: 'Для этой доски сеть ещё не обучена',
    yourTurn: 'Ваш ход', yourWin: 'Вы можете выиграть прямо сейчас', yourBlock: 'Нужно закрыть угрозу',
    thinking: name => `${name} думает…`, wonBy: (name, ply) => `${name} побеждает на ${ply}-м ходу`, youWon: ply => `Вы выиграли на ${ply}-м ходу!`, draw: 'Ничья: свободных клеток не осталось',
    lineShown: kind => `Линия на доске: ${['прямая вдоль оси', 'диагональ слоя', 'диагональ куба', 'диагональ гиперкуба'][kind]}.`,
    lineAcross: kind => `Линия через все кубы: ${['одна и та же клетка в каждом', 'наискосок по одному направлению', 'наискосок по двум направлениям', 'главная диагональ гиперкуба'][kind]}.`, xTurn: 'Ходят крестики', oTurn: 'Ходят нолики',
    netProven: 'Сеть видит форсированную победу.', netLost: 'Сеть видит, что позиция проиграна.',
    netSure: 'Сеть уверена в победе.', netBetter: 'Сеть считает, что стоит лучше.', netEven: 'Сеть считает позицию неясной.', netWorse: 'Сеть считает, что стоит хуже.', netLosing: 'Сеть считает, что проигрывает.',
    fact: (geo, m) => `Клеток ${geo.cells}, линий ${geo.lines.length}.${geo.k !== geo.n ? ' ' + TEXT.ru.need(geo.k) : ''}` + (m ? ` Сеть: итерация ${m.iteration}, ${m.channels}×${m.blocks}.` : ''),
    error: 'Не получилось загрузить сеть.',
    move: 'ход',
  },
  en: {
    title: 'Tic-tac-toe in four dimensions',
    brand: 'Qubic Zero', chartAria: 'Strength of the networks over training iterations', iteration: 'iteration', decimal: '.',
    aboutRulesTitle: 'What the fourth dimension means', ruleCubes: 'Four cubes', ruleCubesText: 'The board is four 4×4×4 cubes. The number of the cube is the fourth coordinate, and the projection above shows them one inside another.',
    ruleLine: 'A line of four', ruleLineText: 'As in ordinary tic-tac-toe, you take every cell of one line. It lies inside one cube or runs through all four, one cell in each.',
    ruleThreat: 'Threats from everywhere', ruleThreatText: 'There are 520 lines, and up to 15 pass through one cell. A game is usually decided after about 14 moves, even with 256 cells.',
    aboutLearnTitle: 'How the network learned to play', aboutLearnText: 'Nobody showed it human games. The network plays against itself and against bots, learns from those games and plays again. A tree search calculates the moves, and the network tells it which moves to look at and who is winning.',
    raceAll: 'Score against the tsp bot, %', raceAllNote: 'tsp looks for forced wins and avoids allowing them. The network moves first in half of the games.', raceSecond: 'Against tsp moving second, %', raceSecondNote: 'The hardest part: on this board the first player has a big advantage.',
    nets: { linenet: 'line network', linenet2: 'new line network', conv4d: '4D convolutions', conv4d_lines: '4D convolutions with line hints', conv2d: 'convolutions', resnet: '3D convolutions', lines: '3D convolutions and lines', hybrid: 'hybrid of convolutions and lines' },
    raceNames: { n44: 'line network, the first', c44: '4D convolutions', c44n: 'convolutions, line hints, new noise', c44s: 'small, before Gumbel', x_b48: 'larger, Gumbel from scratch', x_gl: 'small, Gumbel and league, in this game' },
    aboutResultsTitle: 'Results on all boards', aboutResultsText: 'Every board has its own network. On small boards a solver knows the value of every position, so the result there is exact.',
    colBoard: 'Board', colCells: 'Cells', colLines: 'Lines', colPerCell: 'Lines through a cell', colValue: 'Perfect play', colNet: 'Network', colResult: 'Result',
    valueDraw: 'draw', valueFirst: 'first player wins', vsTsp: 'tsp', vsPure: 'search only', inTraining: 'network still in training', exactResult: (m, n) => `${m} mistakes in ${n.toLocaleString('en')} decisions, checked by a solver`,
    aboutFindingsTitle: 'What the experiments showed',
    find1: 'One line, seven years', find1Text: 'In the first version from 2019 the value network learned not from who won but from whose turn it was. Its loss still fell almost to zero and looked great.',
    find2: 'Bigger does not mean harder', find2Text: '5×5×5 is twice as large as 4×4×4, yet it ends in a draw. Difficulty seems to depend on the number of lines through a cell, not on the size of the board.',
    find3: 'The longest line is the dullest', find3Text: 'On 5×5 with five in a row the second player can always draw: every line has its own pair of cells. This is a proof, not an observation.',
    find4: 'There is no universal network', find4Text: 'On 4×4×4×4 convolutions with line hints played best, while on 9×9 a simple line network won, more than three times smaller than the convolutional one. Direct matches chose the network for each board.',
    readMore: 'More about the training and the findings in the README',
    lead: 'Take all four cells of one line on a 4×4×4×4 board: 256 cells and 520 lines. Your opponent is a neural network that learned by itself, without human games. It all runs in your browser, and if 4D is too much, there are simpler boards, down to 3×3.',
    board: 'Board', crosses: '× crosses, move first', noughts: '○ noughts', strength: 'Network strength', view: 'View',
    hints: 'Highlight threats', thoughts: 'Show what the network thinks', peers: 'Show lines through a cell',
    newGame: 'New game', undo: 'Undo', evaluation: 'network evaluation', spread: 'Distance between layers',
    legendX: 'crosses', legendO: 'noughts', legendXT: '× completes a line here', legendOT: '○ completes a line here',
    howTitle: 'How to play',
    how0: 'On flat boards a line runs along a row, a column or a diagonal. Usually you need all of it, but on large boards a part will do: in gomoku on a 15×15 board you need five in a row, and six or more count too.',
    goal: k => `Get ${k} in a row`, gomoku: 'Gomoku', inRow: k => `${k} in a row`, need: k => `${k} in a row to win.`,
    lineFlat: kind => `The line on the board: ${['along a row or a column', 'a diagonal'][kind]}.`,
    how1: 'To win, take every cell of one line. A cell at the same place in the next layer lies right above or below it, so a line can run through all the layers.',
    how2: 'The strongest weapon is a double threat: your opponent cannot block both. Corners and centres lie on more lines, so they are worth more.',
    how3: 'On the 4×4×4×4 board the fourth dimension is the number of the cube. Cells at the same place in different cubes lie on one line, and a line can also run slantwise through all four cubes. Hover a cell to see every line through it. The Hypercube view shows the same thing as a real projection of a tesseract: a cube inside a cube. Drag to rotate.',
    footer: 'Source code and the story of the project:',
    you: 'You', net: 'Neural network', pure: 'Search without a network', tsp: 'Bot tsp: finds threat sequences', hsp: 'Bot hsp: heuristic', osp: 'Bot osp: simple', random: 'Random moves',
    fast: 'Fast', medium: 'Medium', strong: 'Strong', layers: 'Layers', volume: 'Volume', board4d: 'Table of boards', cubes: 'Four cubes', tesseract: 'Hypercube', cube: 'Cube', rotate4d: 'Rotate in 4D', auto4d: 'auto', note4d: 'The fourth dimension is the number of the cube. A line lies inside one cube or runs through all four, one cell in each.', note4dHyper: 'This is a real projection of a tesseract: every dot is a cell, and one cube lies inside another. Drag to rotate in 3D; hold Shift or use the strip below for the fourth dimension. You can flick it and let go. It holds still while you aim at a cell.', demo4d: ['Example: a straight line inside one cube', 'Example: the same cell in every cube', 'Example: slantwise through all the cubes', 'Example: the main diagonal, all four coordinates change'], cell4d: (w, z, y, x, count) => `Cube ${w}, layer ${z}, row ${y}, cell ${x}. Lines through it: ${count}`,
    layer: 'Layer', noModel: 'The network has not been trained for this board yet',
    yourTurn: 'Your move', yourWin: 'You can win right now', yourBlock: 'You must block a threat',
    thinking: name => `${name} is thinking…`, wonBy: (name, ply) => `${name} wins on move ${ply}`, youWon: ply => `You won on move ${ply}!`, draw: 'A draw: no cells left',
    lineShown: kind => `The line on the board: ${['a straight line along an axis', 'a diagonal of a layer', 'a diagonal of the cube', 'a diagonal of the hypercube'][kind]}.`,
    lineAcross: kind => `A line through all the cubes: ${['the same cell in each', 'slantwise in one direction', 'slantwise in two directions', 'the main diagonal of the hypercube'][kind]}.`, xTurn: 'Crosses to move', oTurn: 'Noughts to move',
    netProven: 'The network sees a forced win.', netLost: 'The network sees the position is lost.',
    netSure: 'The network is sure it wins.', netBetter: 'The network thinks it stands better.', netEven: 'The network finds the position unclear.', netWorse: 'The network thinks it stands worse.', netLosing: 'The network thinks it is losing.',
    fact: (geo, m) => `${geo.cells} cells, ${geo.lines.length} lines.${geo.k !== geo.n ? ' ' + TEXT.en.need(geo.k) : ''}` + (m ? ` Network: iteration ${m.iteration}, ${m.channels}×${m.blocks}.` : ''),
    error: 'Could not load the network.',
    move: 'move',
  },
};

// Сторона, измерения, сколько нужно в ряд
const BOARDS = [[3, 2, 3], [4, 2, 4], [5, 2, 5], [5, 2, 4], [7, 2, 5], [9, 2, 5], [15, 2, 5], [3, 3, 3], [4, 3, 4], [5, 3, 5], [4, 4, 4]];
const AGENTS = ['you', 'net', 'pure', 'tsp', 'hsp', 'osp', 'random'];
const LEVELS = { fast: 25, medium: 80, strong: 300 };

let lang = (() => { try { return localStorage.getItem('lang'); } catch (e) { return null; } })() || ((navigator.language || 'en').startsWith('ru') ? 'ru' : 'en');
let t = TEXT[lang];
let models = {}, geo = Engine.geometry(4, 4), moves = [], state = null;
let players = ['you', 'net'], level = 'fast', view = 'tesseract';
let worker = null, requestId = 0, busy = false, lastInfo = null, shares = null, moveTimer = null;
let cellEls = [], rotX = 62, rotZ = -38, hovered = -1, hyper = null;

function applyText() {
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-t]').forEach(el => { el.textContent = t[el.dataset.t]; });
  $('lang').textContent = lang === 'ru' ? 'English' : 'Русский';
  document.title = 'Qubic Zero';
  drawControls();
  showStatus();
}

function drawControls() {
  $('sizes').innerHTML = [4, 3, 2].map(dim => `<div class="size-group"><span>${dim}D</span>` + BOARDS.filter(b => b[1] === dim).map(([n, d, k]) => {
    const active = n === geo.n && d === geo.d && k === geo.k, size = Array(d).fill(n).join('×');
    const name = n === 15 ? `${t.gomoku} ${size}` : k !== n ? `${size} · ${t.inRow(k)}` : size;
    return `<button data-n="${n}" data-d="${d}" data-k="${k}" aria-pressed="${active}">${name}</button>`;
  }).join('') + '</div>').join('');
  const hasNet = !!models[geo.name];
  for (const [id, side] of [['px', 0], ['po', 1]]) {
    $(id).innerHTML = AGENTS.map(a => `<option value="${a}" ${a === 'net' && !hasNet ? 'disabled' : ''} ${players[side] === a ? 'selected' : ''}>${t[a]}${a === 'net' && !hasNet ? ' — …' : ''}</option>`).join('');
  }
  $('levels').innerHTML = Object.keys(LEVELS).map(k => `<button data-level="${k}" aria-pressed="${level === k}">${t[k]}</button>`).join('');
  const views = geo.d === 2 ? [] : geo.d === 3 ? [['layers', t.layers], ['volume', t.volume]] : [['tesseract', t.tesseract], ['cubes', t.cubes], ['table', t.board4d]];
  if (views.length && !views.some(([k]) => k === view)) view = views[0][0];
  $('views').closest('.field').hidden = !views.length;
  $('views').innerHTML = views.map(([k, label]) => `<button data-view="${k}" aria-pressed="${view === k}">${label}</button>`).join('');
  $('fact').textContent = t.fact(geo, models[geo.name]) + (hasNet ? '' : ' ' + t.noModel + '.');
}

// --- доска ---
function layerHtml(start, title) {
  const n = geo.n;
  let html = `<div>${title ? `<div class="layer-title">${title}</div>` : ''}<div class="layer" style="grid-template-columns:repeat(${n},1fr)">`;
  for (let i = 0; i < n * n; i++) html += `<button class="cell" data-cell="${start + i}"></button>`;
  return html + '</div></div>';
}

function buildBoard() {
  const area = $('board-area'), n = geo.n;
  $('spread-row').hidden = geo.d !== 3 || view !== 'volume';
  $('hyper-row').hidden = geo.d !== 4 || view !== 'tesseract';
  if (hyper) { hyper.destroy(); hyper = null; }
  if (geo.d === 4 && view !== 'table') {
    area.innerHTML = `<p class="hyper-note">${view === 'cubes' ? t.note4d : t.note4dHyper}</p><p class="hyper-caption" id="hyper-caption"></p><canvas id="hyper" aria-label="${view === 'cubes' ? t.cubes : t.tesseract}"></canvas>`;
    cellEls = [];
    hyper = new HyperView($('hyper'), geo, {
      mode: view, auto: $('auto4d').checked,
      labels: { cube: t.cube, demos: t.demo4d, cell: t.cell4d },
      onPlay: cell => { if (humanTurn()) { shares = null; commit(cell); } },
      onCaption: text => { $('hyper-caption').textContent = text; },
      onSpin: spin => {
        $('jog').style.backgroundPositionX = spin + 'px';
        $('jog').setAttribute('aria-valuenow', Math.round((spin * 0.008 * 180 / Math.PI % 360 + 360) % 360));
      },
    });
    drawBoard();
    return;
  }
  if (geo.d === 2) {
    const width = Math.min(640, 200 + 44 * n);
    area.innerHTML = `<div class="boards flat${n >= 9 ? ' dense' : ''}" style="max-width:${width}px">${layerHtml(0, geo.k !== n ? t.goal(geo.k) : '')}</div>`;
  } else if (geo.d === 3 && view === 'volume') {
    area.innerHTML = '<div class="scene" id="scene"><div class="cube" id="cube"></div></div>';
    const cube = $('cube');
    let html = '';
    for (let z = 0; z < n; z++) {
      html += `<div class="slab" data-z="${z}"><div class="layer" style="grid-template-columns:repeat(${n},1fr);grid-template-rows:repeat(${n},1fr)">`;
      for (let i = 0; i < n * n; i++) html += `<button class="cell" data-cell="${z * n * n + i}"></button>`;
      html += '</div></div>';
    }
    cube.innerHTML = html;
    placeVolume();
    attachDrag();
  } else if (geo.d === 3) {
    let html = `<div class="boards" style="grid-template-columns:repeat(auto-fit,minmax(${n >= 5 ? 140 : 130}px,1fr))">`;
    for (let z = 0; z < n; z++) html += layerHtml(z * n * n, `${t.layer} ${z + 1}`);
    area.innerHTML = html + '</div>';
  } else {
    const side = Math.pow(n, geo.d - 2);
    let html = `<div class="boards" style="grid-template-columns:repeat(${n},minmax(112px,1fr));gap:10px;min-width:${n * 122}px">`;
    for (let k = 0; k < side; k++) {
      const w = Math.floor(k / n), z = k % n;
      html += layerHtml(k * n * n, `w${w + 1} · z${z + 1}`);
    }
    area.innerHTML = html + '</div>';
  }
  cellEls = [];
  area.querySelectorAll('.cell').forEach(el => { cellEls[+el.dataset.cell] = el; });
  lineKey = '';
  drawBoard();
}

function cubeSize() { return Math.min($('board-area').clientWidth * 0.46, 370); }
function slabHeight(z) { return (geo.n - 1 - z - (geo.n - 1) / 2) * +$('spread').value * (cubeSize() / 360); }

function placeVolume() {
  const cube = $('cube');
  if (!cube) return;
  const size = cubeSize();
  cube.style.width = cube.style.height = size + 'px';
  cube.style.transform = `rotateX(${rotX}deg) rotateZ(${rotZ}deg)`;
  cube.querySelectorAll('.slab').forEach(slab => {
    slab.style.transform = `translateZ(${slabHeight(+slab.dataset.z)}px)`;
  });
}

function lineKind(line) {
  const a = geo.coords[line[0]], b = geo.coords[line[1]];
  return a.filter((v, i) => v !== b[i]).length - 1;
}

// Луч через куб: две скрещённые полосы, чтобы линия была видна с любого угла
function drawBeam(cube, line) {
  const point = c => {
    const el = cellEls[c];
    return [el.offsetLeft + el.offsetWidth / 2, el.offsetTop + el.offsetHeight / 2, slabHeight(Math.floor(c / (geo.n * geo.n)))];
  };
  const a = point(line[0]), b = point(line[line.length - 1]);
  let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const length = Math.hypot(...d);
  d = d.map(v => v / length);
  const extra = cellEls[line[0]].offsetWidth * 0.75;
  const start = a.map((v, i) => v - d[i] * extra);
  const azimuth = Math.atan2(d[1], d[0]), elevation = Math.asin(d[2]);
  const beam = document.createElement('div');
  beam.className = 'beam';
  beam.style.transform = `translate3d(${start[0]}px,${start[1]}px,${start[2]}px) rotateZ(${azimuth}rad) rotateY(${-elevation}rad)`;
  const total = length + 2 * extra;
  beam.innerHTML = [0, 90].map(angle => `<i style="width:${total}px;transform:rotateX(${angle}deg)"><b></b></i>`).join('');
  cube.appendChild(beam);
}

// Линия поверх плоских досок: проходит через центры выигравших клеток
function drawOverlay(area, line) {
  const box = area.getBoundingClientRect();
  const points = line.map(c => {
    const r = cellEls[c].getBoundingClientRect();
    return [r.left + r.width / 2 - box.left + area.scrollLeft, r.top + r.height / 2 - box.top + area.scrollTop];
  });
  const path = 'M' + points.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'win-overlay');
  svg.setAttribute('width', area.scrollWidth);
  svg.setAttribute('height', area.scrollHeight);
  svg.innerHTML = ['glow', 'core', 'shine'].map(c => `<path class="${c}" pathLength="1" d="${path}"/>`).join('');
  area.appendChild(svg);
}

let lineKey = '', spinning = false;
function stopSpin() { spinning = false; }
function startSpin() {
  if (spinning || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  spinning = true;
  const step = () => {
    if (!spinning || !$('cube')) { spinning = false; return; }
    rotZ += 0.14;
    placeVolume();
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function drawWinLine(force) {
  const area = $('board-area');
  const won = state && state.winner !== null && state.line.length ? state.line : null;
  const key = won ? [geo.name, view, moves.join()].join('|') : '';
  if (key === lineKey && !force) return;
  lineKey = key;
  area.querySelectorAll('.win-overlay, .beam').forEach(el => el.remove());
  const holder = $('cube') || area.querySelector('.boards');
  if (holder) holder.classList.toggle('finished', !!won);
  if (!won) { stopSpin(); return; }
  area.style.setProperty('--win-color', state.winner === 0 ? 'var(--x)' : 'var(--o)');
  if ($('cube')) { drawBeam($('cube'), won); startSpin(); }
  else drawOverlay(area, won);
}

function attachDrag() {
  const scene = $('scene');
  let start = null;
  scene.addEventListener('pointerdown', e => {
    if (e.target.closest('.cell.free')) return;
    stopSpin();
    start = { x: e.clientX, y: e.clientY, rotX, rotZ };
    scene.setPointerCapture(e.pointerId);
  });
  scene.addEventListener('pointermove', e => {
    if (!start) return;
    rotZ = start.rotZ - (e.clientX - start.x) * 0.4;
    rotX = Math.max(0, Math.min(88, start.rotX - (e.clientY - start.y) * 0.4));
    placeVolume();
  });
  const stop = () => { start = null; };
  scene.addEventListener('pointerup', stop);
  scene.addEventListener('pointercancel', stop);
}

function humanTurn() {
  return state && !state.finished && !busy && players[moves.length % 2] === 'you';
}

function drawBoard() {
  if (hyper) {
    if (state) hyper.update({ moves, line: state.line, winner: state.winner, threats: state.threats, shares: $('thoughts').checked ? shares : null,
      canMove: humanTurn(), showPeers: $('peers').checked, showHints: $('hints').checked });
    $('undo').disabled = !moves.length;
    return;
  }
  if (!state || !cellEls.length) return;
  const owner = new Map(moves.map((c, i) => [c, i]));
  const last = moves[moves.length - 1];
  const showHints = $('hints').checked, showHeat = $('thoughts').checked && shares;
  const canMove = humanTurn();
  let top = 0;
  if (showHeat) for (let c = 0; c < geo.cells; c++) top = Math.max(top, shares[c]);
  for (let c = 0; c < geo.cells; c++) {
    const el = cellEls[c];
    const point = geo.coords[c].map(v => v + 1).join(', ');
    if (owner.has(c)) {
      const i = owner.get(c), mark = i % 2 === 0 ? 'x' : 'o';
      const won = state.line.includes(c);
      el.className = `cell ${mark}${won ? ' won' : ''}${c === last && !won ? ' last' : ''}`;
      el.textContent = mark === 'x' ? '×' : '○';
      el.disabled = true;
      el.title = `${t.move} ${i + 1}`;
      el.setAttribute('aria-label', `${point}: ${mark === 'x' ? '×' : '○'}, ${t.move} ${i + 1}`);
    } else {
      const xt = showHints && state.threats[0].includes(c), ot = showHints && state.threats[1].includes(c);
      el.className = 'cell' + (canMove ? ' free' : '') + (xt && ot ? ' both-threat' : xt ? ' x-threat' : ot ? ' o-threat' : '');
      el.disabled = !canMove;
      el.title = '';
      el.setAttribute('aria-label', point);
      el.innerHTML = '';
      if (showHeat && shares[c] > 0.004) {
        const alpha = Math.min(0.85, 0.12 + 0.75 * shares[c] / Math.max(top, 0.05));
        const label = shares[c] >= 0.05 && geo.cells <= 125 ? `<span class="pct">${Math.round(shares[c] * 100)}</span>` : '';
        el.innerHTML = `<span class="heat" style="background:rgba(var(--heat),${alpha.toFixed(2)})"></span>${label}`;
        el.title = `${(shares[c] * 100).toFixed(1)}%`;
      }
    }
  }
  drawPeers();
  $('undo').disabled = !moves.length;
  drawWinLine();
}

function drawPeers() {
  if (hyper) return;
  cellEls.forEach(el => el.classList.remove('peer'));
  if (hovered < 0 || !$('peers').checked) return;
  for (const id of geo.through[hovered]) for (const c of geo.lines[id]) if (c !== hovered) cellEls[c].classList.add('peer');
}

// --- ход игры ---
function agentName(a) { return t[a].split(':')[0]; }

function verdict() {
  if (!lastInfo || lastInfo.value === undefined) return '';
  const v = lastInfo.value;
  if (lastInfo.proven) return v > 0 ? t.netProven : t.netLost;
  return v > 0.6 ? t.netSure : v > 0.2 ? t.netBetter : v > -0.2 ? t.netEven : v > -0.6 ? t.netWorse : t.netLosing;
}

function showStatus() {
  if (!state) return;
  const status = $('status'), sub = $('sub');
  if (state.finished) {
    if (state.winner === null) { status.textContent = t.draw; sub.textContent = ''; }
    else {
      const who = players[state.winner];
      status.textContent = who === 'you' ? t.youWon(moves.length) : t.wonBy(agentName(who), moves.length);
      const across = geo.d === 4 && geo.coords[state.line[0]][0] !== geo.coords[state.line[1]][0];
      sub.textContent = (geo.d === 2 ? t.lineFlat : across ? t.lineAcross : t.lineShown)(lineKind(state.line));
    }
    return;
  }
  const side = moves.length % 2, who = players[side];
  if (who === 'you') {
    status.textContent = state.threats[side].length ? t.yourWin : state.threats[1 - side].length ? t.yourBlock : (players[1 - side] === 'you' ? (side === 0 ? t.xTurn : t.oTurn) : t.yourTurn);
    sub.textContent = verdict();
  } else {
    status.textContent = t.thinking(agentName(who));
    sub.textContent = verdict();
  }
}

function setGauge() {
  if (!lastInfo || lastInfo.value === undefined) { $('gauge').firstElementChild.style.width = '50%'; return; }
  // value дано с точки зрения того, кто сделал ход; шкала показывает шансы крестиков
  const forX = lastInfo.side === 0 ? lastInfo.value : -lastInfo.value;
  $('gauge').firstElementChild.style.width = `${(50 + 50 * forX).toFixed(1)}%`;
}

function ensureWorker() {
  if (!worker) {
    worker = new Worker('worker.js');
    worker.onmessage = onWorkerMessage;
  }
}

function cancelThinking() {
  requestId++;
  clearTimeout(moveTimer);
  if (worker && busy) { worker.terminate(); worker = null; }
  busy = false;
  $('progress').style.visibility = 'hidden';
}

function onWorkerMessage({ data }) {
  if (data.id !== requestId) return;
  if (data.error) {
    busy = false;
    $('progress').style.visibility = 'hidden';
    $('status').textContent = t.error;
    $('sub').textContent = data.error;
    drawBoard();
    return;
  }
  if (data.progress !== undefined) {
    $('progress').firstElementChild.style.width = `${Math.min(100, data.progress * 100)}%`;
    shares = data.shares;
    drawBoard();
    return;
  }
  const { result } = data;
  const started = busyStarted;
  const wait = Math.max(0, MIN_MOVE_MS - (Date.now() - started));
  moveTimer = setTimeout(() => {
    busy = false;
    $('progress').style.visibility = 'hidden';
    if (result.value !== undefined) { lastInfo = { value: result.value, proven: result.proven, side: moves.length % 2 }; setGauge(); }
    shares = result.shares || null;
    commit(result.cell);
  }, wait);
}

const MIN_MOVE_MS = 450;
let busyStarted = 0;

function commit(cell) {
  moves = moves.concat(cell);
  state = Engine.describe(geo, moves);
  if (!state.finished && players[moves.length % 2] === 'you') { /* следующий ход ваш: мысли сети о своём ходе больше не нужны */ }
  advance();
}

function advance() {
  state = Engine.describe(geo, moves);
  drawBoard();
  showStatus();
  if (state.finished) return;
  const who = players[moves.length % 2];
  if (who === 'you') return;
  const id = ++requestId;
  ensureWorker();
  busy = true;
  busyStarted = Date.now();
  shares = null;
  drawBoard();
  showStatus();
  const bar = $('progress');
  bar.firstElementChild.style.width = '0';
  bar.style.visibility = who === 'net' || who === 'pure' ? 'visible' : 'hidden';
  const sims = who === 'pure' ? LEVELS[level] * 8 : LEVELS[level];
  const vcfBudget = geo.cells > 125 ? 60 : 200;
  worker.postMessage({ id, n: geo.n, d: geo.d, k: geo.k, moves, agent: { type: who, sims, vcfBudget, rootSafety: geo.cells > 125 ? 100 : 0 } });
}

function newGame() {
  cancelThinking();
  moves = [];
  lastInfo = null;
  shares = null;
  setGauge();
  state = Engine.describe(geo, moves);
  advance();
}

function undo() {
  cancelThinking();
  if (!moves.length) return;
  let keep = moves.length - 1;
  while (keep > 0 && players[keep % 2] !== 'you' && players.includes('you')) keep--;
  moves = moves.slice(0, keep);
  lastInfo = null;
  shares = null;
  setGauge();
  state = Engine.describe(geo, moves);
  if (players[moves.length % 2] === 'you' || !players.includes('you')) { drawBoard(); showStatus(); if (!players.includes('you')) return; }
  else advance();
}

// --- события ---
$('board-area').addEventListener('click', e => {
  const button = e.target.closest('.cell.free');
  if (!button || !humanTurn()) return;
  shares = null;
  commit(+button.dataset.cell);
});
$('board-area').addEventListener('mouseover', e => {
  const button = e.target.closest('.cell');
  const cell = button ? +button.dataset.cell : -1;
  if (cell !== hovered) { hovered = cell; drawPeers(); }
});
$('board-area').addEventListener('mouseleave', () => { hovered = -1; drawPeers(); });
$('sizes').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  geo = Engine.geometry(+b.dataset.n, +b.dataset.d, +b.dataset.k);
  moves = []; state = null;  // старая партия не должна рисоваться на новой доске
  for (const side of [0, 1]) if (players[side] === 'net' && !models[geo.name]) players[side] = 'tsp';
  drawControls();
  buildBoard();
  newGame();
});
$('levels').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) { level = b.dataset.level; drawControls(); }
});
$('views').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) { view = b.dataset.view; drawControls(); buildBoard(); }
});
$('px').addEventListener('change', e => { players[0] = e.target.value; newGame(); });
$('po').addEventListener('change', e => { players[1] = e.target.value; newGame(); });
$('new').addEventListener('click', newGame);
$('undo').addEventListener('click', undo);
$('spread').addEventListener('input', () => { placeVolume(); drawWinLine(true); });
for (const id of ['hints', 'thoughts', 'peers']) $(id).addEventListener('change', drawBoard);
// Полоска вращения в 4D: в отличие от ползунка, у неё нет концов
let jogFrom = null;
$('jog').addEventListener('pointerdown', e => {
  if (!hyper) return;
  jogFrom = [e.clientX, e.clientY];
  hyper.press();
  $('jog').setPointerCapture(e.pointerId);
});
$('jog').addEventListener('pointermove', e => {
  if (!jogFrom || !hyper) return;
  hyper.moveBy('4d', e.clientX - jogFrom[0], e.clientY - jogFrom[1]);
  jogFrom = [e.clientX, e.clientY];
});
for (const name of ['pointerup', 'pointercancel']) {
  $('jog').addEventListener(name, () => {
    if (jogFrom && hyper) hyper.letGo();
    jogFrom = null;
  });
}
$('jog').addEventListener('keydown', e => {
  const step = { ArrowLeft: [-14, 0], ArrowRight: [14, 0], ArrowUp: [0, -14], ArrowDown: [0, 14] }[e.key];
  if (step && hyper) { e.preventDefault(); hyper.turn('4d', ...step); }
});
$('auto4d').addEventListener('change', e => { if (hyper) hyper.auto = e.target.checked; });
$('lang').addEventListener('click', () => {
  lang = lang === 'ru' ? 'en' : 'ru';
  t = TEXT[lang];
  try { localStorage.setItem('lang', lang); } catch (e) { /* без сохранения */ }
  applyText();
  buildBoard();
  if (typeof drawSite === 'function') drawSite();
});
window.addEventListener('resize', () => { placeVolume(); drawWinLine(true); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (hyper) { hyper.readColors(); hyper.request(); } });

// --- запуск ---
(async () => {
  try {
    const list = await fetch('models/index.json').then(r => r.json());
    models = list;
  } catch (e) { models = {}; }
  // Ссылка может сразу задать партию: ?board=4x4x4x4&x=tsp&o=you&view=volume&lang=en
  const params = new URLSearchParams(location.search);
  const [sides, run] = (params.get('board') || '').split('k');
  const board = sides.split('x').map(Number), wanted = +run || board[0];
  if (BOARDS.some(([n, d, k]) => n === board[0] && d === board.length && k === wanted)) geo = Engine.geometry(board[0], board.length, wanted);
  if (TEXT[params.get('lang')]) { lang = params.get('lang'); t = TEXT[lang]; }
  if (params.get('view')) view = params.get('view');
  if (LEVELS[params.get('level')]) level = params.get('level');
  if (AGENTS.includes(params.get('x'))) players[0] = params.get('x');
  if (AGENTS.includes(params.get('o'))) players[1] = params.get('o');
  applyText();
  for (const side of [0, 1]) if (players[side] === 'net' && !models[geo.name]) players[side] = 'tsp';
  drawControls();
  buildBoard();
  const preset = (params.get('moves') || '').split(',').filter(Boolean).map(Number);
  if (preset.length) {
    moves = preset;
    advance();
  } else {
    newGame();
  }
})();
