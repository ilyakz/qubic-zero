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
