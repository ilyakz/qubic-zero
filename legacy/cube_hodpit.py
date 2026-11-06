import numpy as np
from framework.utils import *
from framework.MCTS import MCTS
import cube_tic_tac_toe_3d_hod.CubeTicTacToeArena as cta
from cube_tic_tac_toe_3d_hod.CubeTicTacToeGame import CubeTicTacToeGame
from cube_tic_tac_toe_3d_hod.CubeTicTacToePlayers import *
from cube_tic_tac_toe_3d_hod.keras.NNet3d import NNetWrapper as nNNet

import os.path

gamesize = 4
countgames = 100
size = 6
BLOCK_SIZE = 20
TILESIZE = 630 / size
ui = dict({
    'width': round(size + 700),
    'height': round(size * TILESIZE)
})

cube = CubeTicTacToeGame(gamesize, ui=ui)

rcp = RandomPlayer(cube).play #random cube player
hcp = HumanCubeTicTacToePlayerUserInterface(cube).play #human cube player
osp = OneStepEndCubePlayer(cube, verbose=False).play
hsp = HeuristicStepCubePlayer(cube).play


n1 = nNNet(cube)
n1.load_checkpoint('./pretrained_models/cube3d-keras/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 180, 'cpuct': 1.0})
mcts1 = MCTS(cube, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


arena = cta.CubeTicTacToeArena(n1p, osp, cube, display = cube.displays)

print(arena.playGames(countgames, verbose=True))


