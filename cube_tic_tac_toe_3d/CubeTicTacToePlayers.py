import numpy as np
import pygame
import sys
from pygame.locals import *


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanCubeTicTacToePlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        #for i in range(len(valid)):
        #    if valid[i]:
        #        print(int((i / self.game.n) / self.game.n), int((i / self.game.n) % self.game.n), int((i % self.game.n) % self.game.n))
        while True:
            a = input()
            z, y, x = [int(x) for x in a.split(' ')]
            if z != -1:
                a = z * self.game.n * self.game.n + y * self.game.n + x 				
            else:
                a = self.game.n ** 3
            if valid[a]:
                break
            else:
                print('Invalid')
       #print("Print A:",a)
        return a


class HumanCubeTicTacToePlayerUserInterface:
    def __init__(self, game):
        self.game = game
        self.tile_size = 40
        pygame.init()

    def get_active_cell(self, mouse_pos):
        """
        :param      mouse_pos:  mouse position on widow screen
        :return:    cell:   cell... just cell
        """
#        for floor in range(self.game.n):
 #           for row in range(self.game.n):
  #              for column in range(self.game.n):
   #                 if (column * self.tile_size) <= mouse_pos[0] <= (column * self.tile_size) + self.tile_size:
    #                    if (floor * self.game.n * self.game.n + row * self.tile_size)  <= mouse_pos[1] <= (floor * self.game.n * self.game.n + row * self.tile_size + self.tile_size):
     #                       cell = (floor, row, column)
      #                      return cell
        #print(self.game.n)
        for floor in range(self.game.n):
            for row in range(self.game.n):
                for column in range(self.game.n):   
                    #print(floor * self.tile_size * self.tile_size, row * self.tile_size)
                    if (column * self.tile_size) <= mouse_pos[0] <= (column * self.tile_size) + self.tile_size:
                        if (floor * self.tile_size * self.game.n + row * self.tile_size) <= mouse_pos[1] <= (floor * self.tile_size * self.game.n + (row * self.tile_size) + self.tile_size):
                            cell = (floor, row, column)
                           # print(mouse_pos[0], mouse_pos[1])
                            #print("MouseclickTo:",floor,row,column)
                            #print(" ")
                            #print(mouse_pos[0], mouse_pos[1])
                            #print(" ")
                            return cell
    @staticmethod
    def exit_game():
        """
        Exit the game.
        """
        pygame.display.quit()
        sys.exit()

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        if valid[-1] == 1:
            z = 6
            x = 0
            y = 0
# выглядит как бред
            if z != -1:
                a = z * self.game.n * self.game.n + y * self.game.n + x 				
            else:
                a = self.game.n ** 3
            return a
        #print(valid)
        for i in range(len(valid)):
            if valid[i]:
               # print(int((i / self.game.n) / self.game.n), int((i / self.game.n) % self.game.n), int((i % self.game.n) % self.game.n))
                pygame.display.update()
        while True:
            act_cell = 0
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.exit_game()
                    elif event.type == MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()
                        act_cell = self.get_active_cell(mouse_pos)
                if act_cell:
                    break
            (z, y, x) = act_cell
            #print("ACT_CELL: ",act_cell)
            if z != -1:
                a = z * self.game.n * self.game.n + y * self.game.n + x 				
            else:
                a = self.game.n ** 3
            if valid[a]:
                break
            else:
                print('Invalid')
        #print("Print A:",a)        
        return a
        

class OneStepEndCubePlayer():
    """ Если остается один шаг до победы - бот прервет победу соперника, либо выиграет сам """
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose
    
    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        #print(enumerate(valid_moves))
        for move, valid in enumerate(valid_moves):
           # print(move," ",valid)
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)
        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))
        return ret_move
    
class HeuristicStepCubePlayer():
    """ Если остается один шаг до победы - бот прервет победу соперника, либо выиграет сам, иначе будет сделан ход в наиболее удачное место """
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose
    
    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        heuristc_move_dict = dict()
        #print(enumerate(valid_moves))
        estimate_moves_pl1, estimate_moves_pl2 = self.game.getEstimatePos(board, self.player_num)
        #print(estimate_moves_pl1)
        #print(estimate_moves_pl2)
        #print(type(estimate_moves_pl1))
        for move, valid in enumerate(valid_moves):
           # print(move," ",valid)
            heuristc_move_dict[move] = estimate_moves_pl1[move] + estimate_moves_pl2[move]
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            #else:
            #    fallback_move_set.add(move)
        
        
        maximum = max(heuristc_move_dict, key=heuristc_move_dict.get)

        #print(maximum, heuristc_move_dict[maximum])
        
        for move, valid in enumerate(valid_moves):
            #print(heuristc_move_dict[maximum], heuristc_move_dict[move])
            if heuristc_move_dict[maximum] == heuristc_move_dict[move]:
                #print(move)
                fallback_move_set.add(move)
        
        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))
        #
        return ret_move
