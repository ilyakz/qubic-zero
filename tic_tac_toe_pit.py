import numpy as np
from framework.utils import *
from framework.MCTS import MCTS
import tic_tac_toe.TicTacToeArena as ttta
from tic_tac_toe.TicTacToeGame import TicTacToeGame
from tic_tac_toe.TicTacToePlayers import *
from tic_tac_toe.tensorflow.NNet import NNetWrapper as nNNet

gamesize = 3
size = 6
BLOCK_SIZE = 20
TILESIZE = 630 / size
ui = dict({
    'width': round(size * TILESIZE + 300),
    'height': round(size * TILESIZE)
})

ttt = TicTacToeGame(gamesize, ui=ui)

rtttp = RandomPlayer(ttt).play
htttp = HumanTicTacToePlayerUserInterface(ttt).play

n1 = nNNet(ttt)
n1.load_checkpoint('./pretrained_models/tic-tac-toe/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
mcts1 = MCTS(ttt, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
n2p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = ttta.TicTacToeArena(n1p, n2p, ttt, display=ttt.displays)
#arena = ttta.TicTacToeArena(n1p, htttp, ttt, display=ttt.displays)
#arena = ttta.TicTacToeArena(rtttp, htttp, ttt, display=ttt.displays)
print(arena.playGames(2, verbose=True))
