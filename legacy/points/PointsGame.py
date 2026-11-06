from __future__ import print_function
from framework.Game import Game
from .PointsLogic import Board
import numpy as np
import pygame
import sys
sys.path.append('..')


class PointsGame(Game):
    """
    Класс Game для точек. Потомок класса Game.
    """
    def __init__(self, n, players=(), ui=False):
        self.score = [0, 0]
        self.n = n
        self.tile_size = 40
        self.black = self.load_picture("./points/images/black.png")
        self.white = self.load_picture("./points/images/white.png")
        self.gray = self.load_picture("./points/images/gray.png")
        self.players = players
        if ui:
            self.ui = ui
            self.surf = pygame.display.set_mode((self.ui['width'], self.ui['height']))
            self.surf.fill((255, 255, 255))

    def getInitBoard(self):
        """
        Возвращает стартовую доску, преобразованную в массив numpy.
        """
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Возвращает кортеж (x,y) с размерностями доски.
        """
        return self.n, self.n

    def getActionSize(self):
        """
        Возвращает количество возможных действий.
        """
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        """
        Возвращает следующее состояние доски.

        :param board: Текущая доска.
        :param player: Текущий игрок.
        :param action:
                    Действие, которое нужно применить.
                    Действие должно быть допустимым ходом.
        :return: следующее состояние доски, после совершение действия.
        """
        if action == self.n * self.n:
            return board, -player
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), int(action % self.n))
        b.execute_move(move, player)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        """
        Возвращает numpy массив с допустимыми ходами для состояния доски
        board и текущего игрока player.

        :param board: Текущее состояние доски.
        :param player: Текущий игрок.
        """
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Определяет, закончилась ли игра. Если закончилась, то с каким результатом.
        :param board: Текущая доска.
        :param player: Текущий игрок.

        :return: 1, если игрок player выиграл,
                 -1, если игрок player проиграл,
                 0, если игра не закончилась
                 1е-4, если ничья.
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        return player*board

    def getSymmetries(self, board, pi):
       # assert(len(pi) == self.n ** 2+1)
       # pi_board = np.reshape(pi[:-1], (self.n, self.n))
       # sym = []
       # for i in range(1, 5):
       #     for j in [True, False]:
       #         newB = np.rot90(board, i)
       #         newPi = np.rot90(pi_board, i)
       #         if j:
       #             newB = np.fliplr(newB)
       #             newPi = np.fliplr(newPi)
       #         sym += [(newB, list(newPi.ravel()) + [pi[-1]])]
       #return sym
        assert(len(pi) == self.getActionSize())  
        pi = np.copy(pi)
        l = [(board, pi)]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    def getCount(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.getCount(player)

    def load_picture(self, filepath):
        picture = pygame.image.load(filepath)
        picture = pygame.transform.scale(picture, (self.tile_size, self.tile_size))
        return picture

    def displays(self, board):
        n = board.shape[0]
        self.surf.fill((255, 255, 255))
        TILESIZE = self.tile_size
        text_high = 100
        for y in range(n):
            for x in range(n):
                piece = board[y][x]
                if piece == 1:
                    picture = self.white
                elif piece == -1:
                    picture = self.black
                else:
                    picture = self.gray
                rect = pygame.Rect(y * TILESIZE, x * TILESIZE, TILESIZE, TILESIZE)
                self.surf.blit(picture, rect)
                pygame.draw.rect(self.surf, (240, 240, 240), rect, 1)
                draw_msg(self.surf, "Белые: " + str(self.getCount(board, 1)), (n * self.tile_size, 0 * text_high))
                draw_msg(self.surf, "Черные: " + str(self.getCount(board, -1)), (n * self.tile_size,
                                                                                 int(0.5 * text_high)))
                draw_msg(self.surf, "Счет " + self.score[0].__str__() + " : " + self.score[1].__str__(),
                         (n * self.tile_size, int(1.5 * text_high)))
                draw_msg(self.surf, "Игрок 1: " + "Черные" if self.players[0] == -1 else "Игрок 1: " + "Белые",
                         (n * self.tile_size, int(3 * text_high)))
                draw_msg(self.surf, "Игрок 2: " + "Черные" if self.players[1] == -1 else "Игрок 2: " + "Белые",
                         (n * self.tile_size, int(3.5 * text_high)))
        pygame.display.update()


def display(board):
    n = board.shape[0]

    for y in range(n):
        print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="")
        for x in range(n):
            piece = board[y][x]
            if piece == -1:
                print("b ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")
    print("   -----------------------")


def draw_msg(surf, text, pos, fontsize=40, color_param=(0, 0, 0), bold=False, erase=False):
    myfont = pygame.font.SysFont("arial", fontsize, bold)
    if erase:
        surf.fill(pygame.Color("white"), (pos[0], pos[1], surf.get_width(), pos[1] + len(text) * fontsize))
    label = myfont.render(text, 1, color_param)
    surf.blit(label, pos)
