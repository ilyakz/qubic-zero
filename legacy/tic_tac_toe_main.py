from framework.Coach import Coach
from tic_tac_toe.TicTacToeGame import TicTacToeGame
#from tic_tac_toe.tensorflow.NNet import NNetWrapper as nn
from tic_tac_toe.keras.NNet import NNetWrapper as nn
#from cube_tic_tac_toe.CubeTicTacToeGame import CubeTicTacToeGame
#from cube_tic_tac_toe.tensorflow.NNet import NNetWrapper as nn
#from cube_tic_tac_toe.keras.NNet import NNetWrapper as nn
#from othello.OthelloGame import OthelloGame
#from othello.tensorflow.NNet import NNetWrapper as nn

from framework.utils import *

args = dotdict({
    'numIters': 1000, 
    'numEps': 25, #было 100
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 5, #50
    'arenaCompare': 50,
    'cpuct': 1,

    'checkpoint': './temp/keras_ttt5/',
    'load_folder_file': ('./temp/keras_ttt5/', 'best.pth.tar'),
    #'checkpoint': './temp/tensorflow_cube/',
    #'load_folder_file': ('./temp/tensorflow_cube/', 'best.pth.tar'),
     
    'load_model': False,
    'numItersForTrainExamplesHistory': 20,
    #'checkpoint': './temp/tensorflow_othello/',
    #'load_folder_file': ('./temp/tensorflow_othello/', 'best.pth.tar'),
    
})

if __name__ == "__main__":
    g = TicTacToeGame(3)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
