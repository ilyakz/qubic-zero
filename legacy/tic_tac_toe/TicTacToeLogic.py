class Board:
    def __init__(self, n):
        # Размер игры . . .
        self.n = n
        # Создаем списковое представление пустой доски
        self.pieces = [None] * self.n
        for i in range(self.n):
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

        for y in range(self.n):
            for x in range(self.n):
                if self[y][x] == color:
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
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x] == color:
                    count += 1
                if self[y][x] == -color:
                    count -= 1
        return count

    def execute_move(self, move, color):
        """
        :param move: tuple(<Int>, <Int>), кортеж, представляющий текущий ход
        :param color: <Int>, 1 для крестиков, -1 для ноликов
        Ставит в клетку с координатами move фигуру color.
        """
        (x, y) = move
        assert self[x][y] == 0
        self[x][y] = color

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def get_legal_moves(self, color):
        moves = set()
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    new_moves = (x,y)
                    moves.add(new_moves)
        return list(moves)
    
    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        win = self.n
        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self[d][d]==color:
                count += 1
        if count==win:
            return True
        count = 0
        for d in range(self.n):
            if self[d][self.n-d-1]==color:
                count += 1
        if count==win:
            return True
        
        return False

    def print_board(self):
        for piece in self.pieces:
            print(piece)
