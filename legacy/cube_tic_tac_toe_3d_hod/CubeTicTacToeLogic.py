class Board:
    def __init__(self, n):
        # Размер игры . . .
        self.n = n
        # Создаем списковое представление пустой доски
        self.pieces =[[[0 for i in range(n)] for j in range(n)] for k in range(n)]
        #print(self.pieces)

    # Добавляем возможность использования оператора взятия индекса ([] и [][])
 #   def __getitem__(self, index):
 #       return self.pieces[index]

    def get_count(self, color):
        """
        :param color: <Int>
                      "цвет" игроков: 1 для крестиков, -1 для ноликов, 0 для незанятых клеток.

        :return: count: <Int>
                        количество фигур типа color.

        Подсчитывает количество фигур типа color на доске.
        """
        count = 0
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][y][x] == color:
                        count += 1
        return count

    def count_diff(self, color):
        """
        :param color: <Int>
                      "цвет" игроков: 1 для крестиков, -1 для ноликов, 0 для незанятых клеток.

        :return: count: <Int>
                        Разница между количеством фигур типа color и -color на доске.

        """
        count = 0
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][y][x] == color:
                        count += 1
                    if self.pieces[z][y][x] == -color:
                        count -= 1
        return count

    def execute_move(self, move, color):
        """
        :param move: tuple(<Int>, <Int>, <Int>), кортеж, представляющий текущий ход
        :param color: <Int>, 1 для крестиков, -1 для ноликов
        Ставит в клетку с координатами move фигуру color.
        """
        (z, y, x) = move
        #print("Board2:",self.pieces)
        self.pieces[z][y][x] = color
    
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()
        for k in range(self.n): #пробегает по оси y с шагом self.n
            for j in range(self.n): #пробегает по оси y
                for i in range(self.n): #пробегает по оси x
                    if self.pieces[k][j][i] == 0:
                        move = (k, j, i)
                        moves.add(move)
        return list(moves)

    def has_legal_moves(self):
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][y][x] == 0:
                        return True
        return False

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        
            
			сумма в строке == ширине строки
			сумма в столбце == высоте столбца
			cумма в глубину == равна глубине кубика
			диагональ побочная
			главная диагональ
			диагональ вниз с верхней строки
			диагональ вниз с нижней строки
			диагональ вглубь с верхнего левого до нижнего правого в одной строке в глубину
			диагональ вниз с верхнего правого до нижнего левого в одной строке в глубину
			диагольналь от верхнего левого до правого нижнего в глубину и в ширину
			диагональ от верхнего правого до левого нижнего в глубину и в ширину
			диагональ от (в верхнем слое) нижнего левого до (нижнего слоя) правого верхнего
            диагональ от (в верхнем слое) нижнего правого до (нижнего слоя) левого верхнего
		"""	
        n = self.n
        if color == 1:
            win = n
        else:
            win = -n
        
        #print(color)
        if any([any([sum([self.pieces[k][j][i] for i in range(n)]) == win for j in range(n)]) == 1 for k in range(n)]) or \
                any([any([sum([self.pieces[k][j][i] for j in range(n)]) == win for i in range(n)]) == 1 for k in range(n)]) or \
                any([any([sum([self.pieces[k][j][i]for k in range(n)]) == win for i in range(n)]) == 1 for j in range(n)]) or \
                any([any([sum([self.pieces[k][i][i] for i in range(n)]) == win]) == 1 for k in range(n)]) or \
                any([any([sum([self.pieces[k][(n - 1 - i)][i] for i in range(n)]) == win]) == 1 for k in range(n)]) or \
                any([any([sum([self.pieces[k][k][i] for k in range(n)]) == win]) == 1 for i in range(n)]) or \
                any([any([sum([self.pieces[k][(n - 1 - k)][i] for k in range(n)]) == win]) == 1 for i in range(n)]) or \
                any([any([sum([self.pieces[k][i][k] for k in range(n)]) == win]) == 1 for i in range(n)]) or \
                any([any([sum([self.pieces[k][i][(n - 1 - k)] for k in range(n)]) == win]) == 1 for i in range(n)]) or \
                sum([self.pieces[k][k][k] for k in range(n)]) == win or \
                sum([self.pieces[k][k][(n - 1 - k)] for k in range(n)]) == win or \
                sum([self.pieces[k][(n - 1 - k)][k] for k in range(n)]) == win or \
                sum([self.pieces[k][(n - 1 - k)][(n - 1 - k)] for k in range(n)]) == win:
                    return True
        return False
        

    def print_board(self):
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    print(self.pieces[z][y][x])
