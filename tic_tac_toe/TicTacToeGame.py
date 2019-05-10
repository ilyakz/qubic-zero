from __future__ import print_function
from tic_tac_toe.TicTacToeLogic import Board
from framework.Game import Game
import numpy as np
import pygame


class TicTacToeGame(Game):
    """
    Game класс для крестиков - ноликов. Потомок класса Game.
    """
    def __init__(self, n, players=(), ui=False):
        self.score = [0, 0]
        self.n = n
        self.tile_size = 105
        self.circle = self.load_picture("./tic_tac_toe/images/circle.bmp")
        self.cross = self.load_picture("./tic_tac_toe/images/cross.bmp")
        self.gray = self.load_picture("./tic_tac_toe/images/empty.png")
        self.players = players
        if ui:
            pygame.init()
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
        Возвращает кортеж (х, у) с размерностями доски.
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
        legal_moves = b.get_legal_moves()
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
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
        if any([sum(b.pieces[i]) == self.n for i in range(self.n)]) or \
                any([sum(b.pieces[:, i]) == self.n for i in range(self.n)]) or \
                sum([b[i][(self.n - 1) - i] for i in range(self.n)]) == self.n or \
                sum(np.diag(b.pieces)) == self.n:
            return 1
        elif any([sum(b.pieces[i]) == -self.n for i in range(self.n)]) or \
                any([sum(b.pieces[:, i]) == -self.n for i in range(self.n)]) or \
                sum([b[i][(self.n - 1) - i] for i in range(self.n)]) == -self.n or \
                sum(np.diag(b.pieces)) == -self.n:
            return -1
        elif any([b.pieces[i, j] == 0 for i in range(self.n) for j in range(self.n)]):
            return 0
        else:
            return 1e-4

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        assert(len(pi) == self.n**2+1)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        li = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                li += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return li

    def stringRepresentation(self, board):
        return board.tostring()

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.count_diff(player)

    def getCount(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.get_count(player)

    def load_picture(self, filepath):
        """
        :param      filepath: path to file
        :return:    picture: picture with size TILESIZE
        """
        picture = pygame.image.load(filepath)
        picture = pygame.transform.scale(picture, (self.tile_size, self.tile_size))
        return picture

    def displays(self, board):
        n = board.shape[0]
 #удалить после проверки    
 #       print("Print board:", n)
        self.surf.fill((255, 255, 255))
        TILESIZE = 105
        for y in range(n):
            for x in range(n):
                piece = board[y][x]
                if piece == 1:
                    picture = self.cross
                elif piece == -1:
                    picture = self.circle
                else:
                    picture = self.gray
                rect = pygame.Rect(y * TILESIZE, x * TILESIZE, TILESIZE, TILESIZE)
                self.surf.blit(picture, rect)
                pygame.draw.rect(self.surf, (240, 240, 240), rect, 1)
                draw_msg(self.surf, "Крестики: " + str(self.getCount(board, 1)),
                         (6 * self.tile_size, 0 * self.tile_size))
                draw_msg(self.surf, "Нолики: " + str(self.getCount(board, -1)),
                         (6 * self.tile_size, int(0.5 * self.tile_size)))
                draw_msg(self.surf, "Счет " + self.score[0].__str__() + " : " + self.score[1].__str__(),
                         (6 * self.tile_size, int(1.5 * self.tile_size)))
                draw_msg(self.surf, "Игрок 1: " + "Нолики" if self.players[0] == -1 else "Игрок 1: " + "Крестики",
                         (6 * self.tile_size, int(3 * self.tile_size)))
                draw_msg(self.surf, "Игрок 2: " + "Нолики" if self.players[1] == -1 else "Игрок 2: " + "Крестики",
                         (6 * self.tile_size, int(3.5 * self.tile_size)))
        pygame.display.update()


def display(board):
    print(board)


def draw_msg(surf, text, pos, fontsize=40, color_param=(0, 0, 0), bold=False, erase=False):
    myfont = pygame.font.SysFont("arial", fontsize, bold)
    if erase:
        surf.fill(pygame.Color("white"), (pos[0], pos[1], surf.get_width(), pos[1] + len(text) * fontsize))
    label = myfont.render(text, 1, color_param)
    surf.blit(label, pos)
