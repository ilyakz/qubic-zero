from framework.Coach import Coach
from cube_tic_tac_toe_3d.CubeTicTacToeGame import CubeTicTacToeGame as Game
#from cube_tic_tac_toe.keras.NNet import NNetWrapper as nn
from cube_tic_tac_toe_3d.keras.NNet3d import NNetWrapper as nn
#from cube_tic_tac_toe.pytorch.NNet import NNetWrapper as nn
#from cube_tic_tac_toe.tensorflow.NNet import NNetWrapper as nn

from framework.utils import *

args = dotdict({
    'numIters': 1000, 
    'numEps': 10, #было 100
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,#50
    'arenaCompare': 40, #50
    'cpuct': 1,

   # 'checkpoint': './temp/tf_cube/',
   # 'load_folder_file': ('./temp/tf_cube/', 'best.pth.tar'),
   # 'checkpoint': './temp/pytorch_cube/',
    #'load_folder_file': ('./temp/pytorch_cube/', 'best.pth.tar'),
    'checkpoint': './temp/keras_cube/',
    'load_folder_file': ('./temp/keras_cube/', 'best.pth.tar'),
    
    'numItersForTrainExamplesHistory': 20,
    'load_model': False,
})

if __name__ == "__main__":

    g = Game(3)

    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
