import numpy as np
from framework.utils import *
from framework.MCTS import MCTS
import framework.Arena as ttta
from tic_tac_toe.TicTacToeGame import TicTacToeGame
from tic_tac_toe.TicTacToePlayers import *
from tic_tac_toe.keras.NNet import NNetWrapper as nNNet

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
osp = OneStepEndPlayer(ttt).play

n1 = nNNet(ttt)
n1.load_checkpoint('./pretrained_models/ttt-keras/', 'best2.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
mcts1 = MCTS(ttt, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

#n2 = nNNet(ttt)
#n2.load_checkpoint('./pretrained_models/ttt5-keras/', 'bestold.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
#mcts2 = MCTS(ttt, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#arena = ttta.TicTacToeArena(n1p, n2p, ttt, display=ttt.displays)
arena = ttta.TicTacToeArena(n1p, osp, ttt, display=ttt.displays)
#arena = ttta.TicTacToeArena(n1p, htttp, ttt, display=ttt.displays)
#arena = ttta.TicTacToeArena(rtttp, htttp, ttt, display=ttt.displays)
print(arena.playGames(20, verbose=True))
