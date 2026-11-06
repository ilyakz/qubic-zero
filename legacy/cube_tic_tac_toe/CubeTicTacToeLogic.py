class Board:
    def __init__(self, n):
        # Размер игры . . .
        self.n = n
        # Создаем списковое представление пустой доски
        self.pieces = [None] * self.n * self.n
        for i in range(self.n * self.n):
            self.pieces[i] = [0] * self.n

    # Добавляем возможность использования оператора взятия индекса ([] и [][])
    def __getitem__(self, index):
        return self.pieces[index]

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
                    if self[z * self.n + y][x] == color:
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
                    if self[z * self.n + y][x] == color:
                        count += 1
                    if self[z * self.n + y][x] == -color:
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
        self.pieces[z * self.n + y][x] = color
    
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()
        for k in range(self.n): #пробегает по оси y с шагом self.n
            for j in range(self.n): #пробегает по оси y
                for i in range(self.n): #пробегает по оси x
                    if self.pieces[k * self.n + j][i] == 0:
                        move = (k, j, i)
                        moves.add(move)
        return list(moves)

    def has_legal_moves(self):
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z * self.n + y][x] == 0:
                        return True
        return False

    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        """
        не переделал
        """
        return 0
        

    def print_board(self):
        for piece in self.pieces:
            print(piece)
