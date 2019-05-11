from framework.Coach import Coach
from cube_tic_tac_toe.CubeTicTacToeGame import CubeTicTacToeGame
from cube_tic_tac_toe.keras.NNet import NNetWrapper as nn
from framework.utils import *

args = dotdict({
    'numIters': 1000, 
    'numEps': 10, #было 100
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 10,#50
    'arenaCompare': 10, #50
    'cpuct': 1,

    'checkpoint': './temp/keras_cube/',
    'load_folder_file': ('./temp/keras_cube/', 'best.pth.tar'),

    'load_model': False,

})

if __name__ == "__main__":

    g = CubeTicTacToeGame(4)

    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()
