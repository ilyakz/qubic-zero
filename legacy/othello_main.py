from framework.Coach import Coach

from othello.OthelloGame import OthelloGame
from othello.keras.NNet import NNetWrapper as nn
#from othello.pytorch.NNet import NNetWrapper as nn

from framework.utils import *

args = dotdict({
    'numIters': 1000, 
    'numEps': 50, #было 100
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 50,
    'cpuct': 1,

   
    'load_model': False,
    #'checkpoint': './temp/tensorflow_othello/',
    #'load_folder_file': ('./temp/tensorflow_othello/', 'best.pth.tar'),
    'checkpoint': './temp/keras_othello/',
    'load_folder_file': ('./temp/keras_othello/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    
})

if __name__ == "__main__":

    g = OthelloGame(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()
