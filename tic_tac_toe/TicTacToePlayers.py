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


class HumanTicTacToePlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            a = input()
            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')
        return a


class HumanTicTacToePlayerUserInterface:
    def __init__(self, game):
        self.game = game
        self.tile_size = 105
        pygame.init()

    def get_active_cell(self, mouse_pos):
        """
        :param      mouse_pos:  mouse position on widow screen
        :return:    cell:   cell... just cell
        """
        for row in range(self.game.n):
            for column in range(self.game.n):
                if (column * self.tile_size) <= mouse_pos[0] <= (column * self.tile_size) + self.tile_size:
                    if (row * self.tile_size) <= mouse_pos[1] <= (row * self.tile_size) + self.tile_size:
                        cell = (row, column)
                        print(" ")
                        print(row, column)
                        print(" ")
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
            x = 6
            y = 0
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            return a
        print(valid)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
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
            (y, x) = act_cell
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')
        return a
        
class OneStepEndPlayer():
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
