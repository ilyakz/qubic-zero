from __future__ import print_function
from cube_tic_tac_toe_3d.CubeTicTacToeLogic import Board
from framework.Game import Game
import numpy as np
import pygame
import math


class CubeTicTacToeGame(Game):
    """
    Game класс для кубических крестиков - ноликов. Потомок класса Game.
    """
    def __init__(self, n, players=(), ui=False):
        self.score = [0, 0]
        self.n = n
        self.tile_size = 40
        self.circle = self.load_picture("./cube_tic_tac_toe/images/circle.bmp")
        self.cross = self.load_picture("./cube_tic_tac_toe/images/cross.bmp")
        self.gray = self.load_picture("./cube_tic_tac_toe/images/empty.png")
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
        #print(b.pieces)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Возвращает кортеж (z , y, x) с размерностями доски.
        """
        return (self.n, self.n, self.n)
      

    def getActionSize(self):
        """
        Возвращает количество возможных действий.
        """
        return self.n * self.n * self.n + 1

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
        if action == self.n * self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int((action / self.n) / self.n), int((action / self.n) % self.n), int((action % self.n) % self.n))
        #print("MOVE:", move)
        b.execute_move(move, player)
        return (b.pieces, -player)

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
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for z, y, x in legal_moves:
            valids[self.n * self.n * z + self.n * y + x] = 1
        return np.array(valids)
    
    def getEstimatePos(self, board, player):
        """
			1.сумма в строке 
			2.сумма в столбце 
			3.cумма в глубину 
			4.диагональ побочная
			5.главная диагональ
			6.диагональ вниз с верхней строки
			7.диагональ вниз с нижней строки
			8.диагональ вглубь с верхнего левого до нижнего правого в одной строке в глубину
			9.диагональ вниз с верхнего правого до нижнего левого в одной строке в глубину
			10.диагольналь от верхнего левого до правого нижнего в глубину и в ширину
			11.диагональ от верхнего правого до левого нижнего в глубину и в ширину
			12.диагональ от (в верхнем слое) нижнего левого до (нижнего слоя) правого верхнего
            13.диагональ от (в верхнем слое) нижнего правого до (нижнего слоя) левого верхнего
        """	
        b = Board(self.n)
        b.pieces = np.copy(board)
        n = self.n

        
        valid_moves_pl1 = [0]*self.getActionSize()
        valid_moves_pl2 = [0]*self.getActionSize()
        #print(player)
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    if b.pieces[z][y][x] == 0:
                        valid_moves_pl1[z * n * n  + y * n + x ] = +(sum([b.pieces[z][j][x] == player for j in range(n)]) + \
                            sum([b.pieces[z][y][i] == player for i in range(n)]) + \
                            sum([b.pieces[k][y][x] == player for k in range(n)]) + \
                            ((n - 1 - y) == x) * sum([b.pieces[z][(n - 1 - i)][i] == player for i in range(n)]) +\
                            (y == x) * sum([b.pieces[z][i][i] == player for i in range(n)]) + \
                            (z == y) * sum([b.pieces[k][k][x] == player for k in range(n)]) + \
                            (z == (n - 1 - y)) * sum([b.pieces[k][(n - 1 - k)][x] == player for k in range(n)]) + \
                            (z == x) * sum([b.pieces[k][y][k] == player for k in range(n)]) + \
                            (z == (n - 1 - x)) * sum([b.pieces[k][y][n - 1 - k] == player for k in range(n)]) + \
                            (z == y == x) * sum([b.pieces[k][k][k] == player for k in range(n)]) + \
                            (z == y == (n - 1 - x)) * sum([b.pieces[k][k][n - 1 - k] == player for k in range(n)]) + \
                            (z == (n - 1 - y) == x) * sum([b.pieces[k][(n - 1 - k)][k] == player for k in range(n)]) + \
                            (z == (n - 1 - y) == (n - 1 - x)) * sum([b.pieces[k][(n - 1 - k)][n - 1 - k] == player for k in range(n)]))
                        valid_moves_pl2[z * n * n  + y * n + x ] = +(sum([b.pieces[z][j][x] == -player for j in range(n)]) + \
                            sum([b.pieces[z][y][i] == -player for i in range(n)]) + \
                            sum([b.pieces[k][y][x] == -player for k in range(n)]) + \
                            ((n - 1 - y) == x) * sum([b.pieces[z][(n - 1 - i)][i] == -player for i in range(n)]) +\
                            (y == x) * sum([b.pieces[z][i][i] == -player for i in range(n)]) + \
                            (z == y) * sum([b.pieces[k][k][x] == -player for k in range(n)]) + \
                            (z == (n - 1 - y)) * sum([b.pieces[k][(n - 1 - k)][x] == -player for k in range(n)]) + \
                            (z == x) * sum([b.pieces[k][y][k] == -player for k in range(n)]) + \
                            (z == (n - 1 - x)) * sum([b.pieces[k][y][n - 1 - k] == -player for k in range(n)]) + \
                            (z == y == x) * sum([b.pieces[k][k][k] == -player for k in range(n)]) + \
                            (z == y == (n - 1 - x)) * sum([b.pieces[k][k][n - 1 - k] == -player for k in range(n)]) + \
                            (z == (n - 1 - y) == x) * sum([b.pieces[k][(n - 1 - k)][k] == -player for k in range(n)]) + \
                            (z == (n - 1 - y) == (n - 1 - x)) * sum([b.pieces[k][(n - 1 - k)][n - 1 - k] == -player for k in range(n)]))
                    else:
                        continue
        
        #print(valid_moves_pl2)
        #v = [1] * len(board)
        #return [(board,v)]
        return (valid_moves_pl1, valid_moves_pl2)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        #print(b.pieces)
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

   
    
    def getCanonicalForm(self, board, player):
        #print(board)
        return player * board

    def getSymmetries(self, board, pi):
        assert(len(pi) == self.getActionSize())  
        pi = np.copy(pi)
        #l = [(board, pi)]
        #return l
        #return [(board, pi), (board[:,::-1,], pi[::-1])]
        
        pi_board = np.reshape(pi[:-1], (self.n,self.n, self.n))
       # print(pi)
        li = []
        rot1 = {(1,0),(2,0)}
        for k in rot1:
            for i in range(1, 5):
                for m in range(1, 5):
                    for j in [True, False]:
                        newB = np.rot90(np.rot90(board, m, (1,-1)), i, k)
                        newPi = np.rot90(np.rot90(pi_board, m, (1,-1)), i, k)
                        if j:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        li += [(newB, list(newPi.ravel()) + [pi[-1]])]
        
        
        #print(li)
       
        return li


    def stringRepresentation(self, board):
        #print(board)
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
        #n = board.shape[0]
        n = self.n
        self.surf.fill((255, 255, 255))
        TILESIZE = self.tile_size
        for x in range(n):
            for y in range(n):
                for z in range(n):				
                    piece = board[z][y][x]
                    
                    if piece == 1:
                        picture = self.cross
                    elif piece == -1:
                        picture = self.circle
                    else:
                        picture = self.gray
                    rect = pygame.Rect(x * TILESIZE, z * TILESIZE * n + y * TILESIZE, TILESIZE, TILESIZE)
                    
                    self.surf.blit(picture, rect)
                    pygame.draw.rect(self.surf, (240, 240, 240), rect, 1)
                    
                    if z != 0:
                        pygame.draw.line(self.surf, (255, 160, 0), (0, TILESIZE * n * z ), (TILESIZE * n,TILESIZE * n * z), 5)
						
                    draw_msg(self.surf, "Крестики: " + str(self.getCount(board, 1)),
                         (6 * self.tile_size, 0 * self.tile_size))
                    draw_msg(self.surf, "Нолики: " + str(self.getCount(board, -1)),
                         (6 * self.tile_size, int(1 * self.tile_size)))
                    draw_msg(self.surf, "Счет " + self.score[0].__str__() + " : " + self.score[1].__str__(),
                         (6 * self.tile_size, int(2 * self.tile_size)))
                    draw_msg(self.surf, "Игрок 1: " + "Нолики" if self.players[0] == -1 else "Игрок 1: " + "Крестики",
                         (6 * self.tile_size, int(4 * self.tile_size)))
                    draw_msg(self.surf, "Игрок 2: " + "Нолики" if self.players[1] == -1 else "Игрок 2: " + "Крестики",
                         (6 * self.tile_size, int(5 * self.tile_size)))
        pygame.display.update()


def display(board):
    print(board)


def draw_msg(surf, text, pos, fontsize=40, color_param=(0, 0, 0), bold=False, erase=False):
    myfont = pygame.font.SysFont("arial", fontsize, bold)
    if erase:
        surf.fill(pygame.Color("white"), (pos[0], pos[1], surf.get_width(), pos[1] + len(text) * fontsize))
    label = myfont.render(text, 1, color_param)
    surf.blit(label, pos)
