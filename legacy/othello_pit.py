import numpy as np
from framework.utils import *
import othello.OthelloArena as oa
from framework.MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.tensorflow.NNet import NNetWrapper as nNNet

size = 6
BLOCK_SIZE = 20
TILESIZE = 630 / size
ui = dict({
    'width': round(size * TILESIZE + 300),
    'height': round(size * TILESIZE)
})

g = OthelloGame(6, ui=ui)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
hp1 = HumanPlayerUserInterface(g).play

#nnet players
n1 = nNNet(g)
n1.load_checkpoint('./pretrained_models/othello/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 6, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# nnet players
#n1 = nNNet(g)
#n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
#args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# m1 = nNNet(g)
# m1.load_checkpoint('./pretrained_models/tensorflow/', 'best.pth.tar')
# args2 = dotdict({'numMCTSSims': 6, 'cpuct': 1.0})
# mcts2 = MCTS(g, m1, args2)
# n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#arena = oa.OthelloArena(n1p, hp1, g, display=g.displays)
arena = oa.OthelloArena(n1p, hp1, g, display=g.displays)
#arena = oa.OthelloArena(gp, hp1, g, display=g.displays)
print(arena.playGames(6, verbose=True))
