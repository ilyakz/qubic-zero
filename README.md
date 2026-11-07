<div align="center">

# Qubic Zero

**Tic-tac-toe in four dimensions, against a neural network that taught itself.**

[![Play in your browser](https://img.shields.io/badge/%E2%96%B6_Play_in_your_browser-6250d6?style=for-the-badge)](https://ilyakz.github.io/qubic-zero/)

[![Tests](https://github.com/ilyakz/qubic-zero/actions/workflows/tests.yml/badge.svg)](https://github.com/ilyakz/qubic-zero/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2-ee4c2c)
![Runs in the browser](https://img.shields.io/badge/runs_in-the_browser-1baf7a)

[Русская версия](README.ru.md)

<img src="docs/img/hero.gif" width="640" alt="A rotating projection of the 4×4×4×4 board with a winning line through all four dimensions">

*A real projection of the 4×4×4×4 board: 256 cells, 520 lines. Crosses have just won along the main diagonal of the hypercube.*

</div>

Take all four cells of one line on a 4×4×4×4 board, before a network does it first. No installation: the network, the search and the rules all run on the page, and it answers in 2–5 seconds on an ordinary laptop. If 4D is too much, there are simpler boards, from 3×3 up to gomoku.

| | | |
|:-:|:-:|:-:|
| ![The hypercube view](docs/img/game_hypercube.png) | ![Four cubes in a row](docs/img/game_cubes.png) | ![Gomoku on a 15×15 board](docs/img/game_gomoku.png) |
| **Hypercube:** spin it in four dimensions | **Four cubes:** the number of the cube is the fourth coordinate | **Gomoku:** one of the eleven boards |

## Contents

[The 4D game](#the-4d-game) · [What was wrong in 2019](#what-was-wrong-in-2019) · [All boards](#all-boards) · [Findings](#findings) · [Run it yourself](#run-it-yourself) · [How it works](#how-it-works) · [Credits](#credits)

## The 4D game

The board is four 4×4×4 cubes. A line lies inside one cube or runs through all four, one cell in each, and up to 15 lines pass through a cell. Threats appear from everywhere, so a game is usually decided after about 14 moves, even with 256 cells.

The network in the game has 0.7 million weights and a 2.8 MB file. Here is how the candidates did, each at its best stretch of training, in 200 games per opponent, with the settings the browser uses:

| Network | Weights | Against `tsp` | Moving second | Search only | Direct match with the small network |
|---|---:|---:|---:|---:|---:|
| **Small 4D convolutions + line hints, in the game** | **0.7M** | **69%** | **38%** | **99%** | — |
| 4D convolutions | 4.0M | 85.5% | 71% | 98% | 69–131 |
| 4D convolutions + line hints + tuned noise | 4.0M | 78% | 60% | 99.5% | 88–112 |
| 4D convolutions + line hints | 4.0M | 73.5% | 47% | 99.5% | 72–128 |
| New line network | 0.5M | 52% | 34% | 99% | not played |
| The first line network | 0.1M | 13.5% | 2% | 93% | 47–153 |

The largest network is the strongest against `tsp`, yet the small one beats it head to head, and it answers about four times faster. That is why it is in the game. At the Fast level (25 simulations) it still scores 73% against `tsp`, in 100 games.

**What this does not prove.** `tsp` was also a sparring partner in half of the training games, so the scores against it may flatter the networks. The first player wins most games between strong networks, so moving second is the real test, and even the best result here is 71%. I do not know who wins this board with perfect play. I also did not find another AlphaZero-style network trained for 4×4×4×4, but a search can miss things, so I will not claim to be the first.

![The race between networks on the 4×4×4×4 board](docs/img/race_4d.png)

*Scores from checks during training, averaged over three checks. Each check is only 40 games.*

**The opponents.** `osp` and `hsp` are the bots from 2019, which beat the old network. `tsp` is new: it searches for forced wins and avoids allowing them. "Search only" knows the same rules and threat sequences as the network but never learned anything.

## The story

In May 2019 I wrote an AlphaZero for 4×4×4 tic-tac-toe on Keras and TensorFlow 1. It never played well: 60% against a bot that moves randomly unless it can win or block, and 35% against a simple heuristic. I could not find out why. The [commit history](https://github.com/ilyakz/qubic-zero/commits/master/) of this repository still starts there, on 10 May 2019, and the code from that time is kept in [`legacy/`](legacy/).

Seven years later I came back to the code with one goal: find the error and finish the job. It turned out to be a single line, and the training data saved in 2019 proved it.

## What was wrong in 2019

The game returned the result from the point of view of the crosses, while the trainer expected it from the point of view of the current player:

```python
# framework/Coach.py expects "+1 if `player` won" and flips the sign for the other side
r = self.game.getGameEnded(board, self.curPlayer)
return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

# the game ignored `player`: +1 whenever crosses had a line, -1 whenever noughts had one
def getGameEnded(self, board, player): ...
```

Together this gives **-1 to every position of the crosses and +1 to every position of the noughts, no matter who won.** All 11,376 examples in the saved file are labelled exactly like that. The value network learned to tell whose turn it is, and the search trusted it.

What made it so hard to see: the loss looked wonderful. Whose turn it is can be read straight off the board, so the value loss fell almost to zero. I reproduced the bug in a control run:

![Value loss of the broken run falls to zero, while it predicts the winner worse than a coin](docs/img/bug_2019.png)

The rules, the 76 winning lines, the 48 symmetries and the move indexing from 2019 were all correct: I checked them against brute force.

## All boards

Every board has its own network, and direct matches chose its design. On small boards the check is exact: a solver knows the true value of every position, so a mistake cannot hide.

| Board | In a row | Perfect play | Network | How it was checked | Result |
|---|:-:|---|---|---|---|
| **4×4×4×4** | 4 | not known to me | 4D convolutions 32×4 | 200 games per bot | 69% `tsp`, 99% search only |
| 3×3 | 3 | draw | lines 32×2 | solver, **every** one of the 4,520 possible positions | **0 mistakes** |
| 4×4 | 4 | draw | lines 32×3 | solver, every reply: 343,790 decisions, plus 20,003 random positions | **0 mistakes** |
| 5×5 | 5 | draw, proved [below](#findings) | lines 32×3 | 100 games per bot | never lost; draws with `tsp`, `hsp` and search only |
| 5×5 | 4 | not known to me | lines 48×3 | 100 games per bot | never lost; 62.5% `tsp`, 75% search only |
| 7×7 | 5 | not known to me | lines 64×4 | 100 games per bot | never lost; 58.5% `tsp`, 74% search only |
| 9×9 | 5 | not known to me | lines 64×4 | 100 games per bot | 100% against every bot |
| Gomoku 15×15 | 5 | first player wins ([Allis, 1994](https://en.wikipedia.org/wiki/Gomoku)) | lines 64×4 | 100 games per bot | 92.5% `tsp`, 100% search only |
| 3×3×3 | 3 | first player wins | lines 48×3 | solver, every reply: 372,613 decisions | **0 mistakes** |
| 4×4×4 | 4 | first player wins ([Patashnik, 1980](https://en.wikipedia.org/wiki/3D_tic-tac-toe)) | lines 96×6 | 300 games per bot at 800 simulations | 100% `tsp`, 98.7% `hsp`, 90.8% search only |
| 5×5×5 | 5 | not known to me | lines 64×4 | 60 games per bot | draws with `tsp`, 98.3% search only |

**On 9×9 and gomoku the bots have become too weak to measure much.** A hundred percent means the network beats them, not that it plays gomoku well. A serious gomoku engine would be a much harder test, and I have not run one.

Part of the strength comes from rules rather than learning: winning moves, forced blocks and threat sequences are found by a small C routine, not by the network. I kept that because it makes the network learn fast, but it means this is AlphaZero plus game knowledge, not pure AlphaZero.

## Findings

**There is no universal network.** On 4×4×4×4 convolutions beat the line network by 153–47. On 9×9 it went the other way: the line network beat convolutions 3.5 times its size by 48–18, with 134 draws. On 4×4×4 they were level, 98–11–91. So each board got a network chosen by direct matches.

**Search noise has to fit the board.** I had kept the noise from small boards, and on 256 moves it was nearly uniform. With noise tuned to the number of moves and a longer random start of the game, the same network scored 78% instead of 73.5% against `tsp`, and 60% instead of 47% moving second. It also got there in far fewer iterations.

**A lower learning rate paid off late.** Convolutions stayed around 70% against `tsp` for dozens of iterations and then reached about 80% after the rate was lowered again. Stopping at the first plateau would have missed that, so the plateau rule now lowers the rate before it gives up.

**A check inside the search gave the biggest jump.** With the same weights and the same 100 simulations, checking every new position for a forced win lifted the score against `tsp` from 47% to 90%. Moved to C, it even made the search faster, because the network is asked almost half as often.

![Same weights, same simulations](docs/img/tree_check.png)

**A check at the root makes the difference visible.** With more moves than simulations, the search cannot even look at each move, so a check removes moves that allow a forced win. Without it, direct matches between 4D networks were close to even, and with it the new networks beat the old one by 153–47.

**A bigger board can be an easier game.** 5×5×5 is twice as large as 4×4×4, yet games there fill 99% of the board and end in a draw. What seems to matter is how many lines pass through a cell:

| Board | Cells | Lines through a cell | Board filled when decided | Self-play |
|---|---:|---:|---:|---|
| 4×4 | 16 | 2.5 | 100% | draws |
| 5×5×5 | 125 | 4.4 | 99% | 98% draws |
| 4×4×4 | 64 | 4.8 | 17% | first player wins 98% |
| 4×4×4×4 | 256 | 8.1 | 5% | first player wins almost always |

**The longest line is the dullest.** On 5×5 with five in a row there are only 12 lines. Each line can be given its own pair of cells, with no pair shared: 24 cells are enough, and the board has 25. Whenever the first player takes one cell of a pair, the second takes the other, so the first player can never complete a line. This proves a draw without any search. The same works on 6×6 and 7×7. A shorter line on a larger board is what makes the game interesting again.

**Two traps I fell into myself.**
- I read a value accuracy of 82% as proof that the network understands positions. Then I noticed it equals the share of first-player wins almost exactly: the network was mostly betting on the first player. Always compare with the simplest possible rule.
- On 3×3 my dashboard showed the network losing 14 of 20 games as the second player, which looked terrible. The cause was my check: matches started with two random moves, and the solver shows that in 48 of the 72 such openings the second player has already lost. From an empty board the same network drew 40 of 40.

**What did not help, measured in direct matches:** random openings in self-play (149–4–147), propagating proofs up the tree (52–48), making near wins worth more (96–6–98). A hybrid that passes messages along lines inside every convolutional block reached only 18% against `tsp` after 30 iterations, so I stopped it early. Deeper search at play time helps little: on 4×4×4×4 going from 100 to 1600 simulations added about 10 points against `tsp`, and a bigger threat search budget added nothing.

## Run it yourself

```bash
pip install -r requirements.txt     # also needs gcc, a small part of the rules is written in C
python play.py                      # the game at http://127.0.0.1:8766
python train.py --run my_run --n 4 --d 4 --arch conv4d_lines --channels 32 --blocks 4 --league tsp=0.5 --noise-alpha 0.04 --temperature-plies 12 --patience 6 --lr-drops 2
python dashboard/server.py          # live charts at http://127.0.0.1:8765
python -m unittest discover -s tests && node tests/test_web.js
```

Other boards: `--n 4 --d 3` for 4×4×4, `--n 15 --d 2 --k 5` for gomoku. `tools/solve_small.py` runs the exact check on small boards, `compare.py` plays networks against each other, `tools/final_match.py` plays them against the bots.

The dashboard shows all runs at a glance, compares networks on one board and tracks the load on the machine.

![Training dashboard](docs/img/dashboard.png)

## How it works

| File | What is inside |
|---|---|
| `qubic/game.py` | rules for any board size, dimension and line length: lines, symmetries, bitboards |
| `qubic/fast.c` | the part of the rules used on every search node: wins, threats, threat sequences |
| `qubic/mcts.py` | PUCT search; the leaves of all games go to the GPU in one batch |
| `qubic/net.py` | line networks, convolutions in 2D, 3D and 4D, and the 2019 network moved to PyTorch one to one |
| `qubic/play.py`, `qubic/worker.py` | self-play and matches in parallel processes |
| `qubic/bots.py` | the opponents |
| `train.py` | the training loop: games against bots, a lower rate on a plateau, then an automatic stop |
| `dashboard/` | live dashboard in two languages |
| `docs/` | the game: rules, search and networks in plain JavaScript, checked against PyTorch to 1e-6 |
| `legacy/` | the code as it was in 2019 |

PyTorch has no 4D convolution, so `qubic/net.py` builds one out of 3D convolutions, and a test checks it against the definition cell by cell.

The search and the training loop follow AlphaZero. The main additions: the leaves are batched across all games, games stop as soon as a forced win is proven, and repeated positions are not evaluated twice.

## Credits

This project started from [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair and contributors: the 2019 code in `legacy/` is built on it. Everything else was written from scratch and is under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for the details.

I also proposed the 4D game to alpha-zero-general itself: [pull request #338](https://github.com/suragnair/alpha-zero-general/pull/338), with the rules, the network and the trained weights.

Thanks to [Anthropic](https://www.anthropic.com) and [Claude Code](https://claude.com/claude-code) with the Claude Opus 5.5 model, which helped me finish this old project and reach these results.
