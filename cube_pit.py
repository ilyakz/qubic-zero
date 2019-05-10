import numpy as np
from framework.utils import *
from framework.MCTS import MCTS
import cube_tic_tac_toe.CubeTicTacToeArena as cta
from cube_tic_tac_toe.CubeTicTacToeGame import CubeTicTacToeGame
from cube_tic_tac_toe.CubeTicTacToePlayers import *
from cube_tic_tac_toe.keras.NNet import NNetWrapper as nNNet
from cube_tic_tac_toe.keras_3.NNet import NNetWrapper as nNNet2
import os.path

gamesize = 4
countgames = 8
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
osp = OneStepEndCubePlayer(cube).play
hsp = HeuristicStepCubePlayer(cube).play

n1 = nNNet2(cube)
#n1.load_checkpoint('./pretrained_models/cube-keras/', 'best.pth.tar')
n1.load_checkpoint('./pretrained_models/cube-keras/', 'checkpoint3_2.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(cube, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = nNNet(cube)
#n2.load_checkpoint('./pretrained_models/cube-keras-2/', 'best_i10.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts2 = MCTS(cube, n1, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


#arena = cta.CubeTicTacToeArena(hcp, hsp, cube, display = cube.displays)
#arena = cta.CubeTicTacToeArena(hcp, rcp, cube, display = cube.displays)
#arena = cta.CubeTicTacToeArena(n1p, n2p, cube, display = cube.displays)

#print(arena.playGames(countgames, verbose=True))

print("random VS all")

for i in range(1,31):
    n2 = nNNet(cube)
    if(os.path.exists('./pretrained_models/cube-keras/checkpoint_'+str(i)+'.pth.tar')):
        n2.load_checkpoint('./pretrained_models/cube-keras/', 'checkpoint_'+str(i)+'.pth.tar')
    else:
        continue
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(cube, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    
    arena = cta.CubeTicTacToeArena(n1p, n2p, cube, display = cube.displays)
    #print("cp_5 VS cp_" + str(i))
    print(i, end = ' ')
    print(arena.playGames(10, verbose=False))
