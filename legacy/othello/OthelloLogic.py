class Board:

    # Направления для поиска
    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, n):
        """
        Настраиваем стартовое состояние доски.
        :param n: Размер доски.
        """
        self.n = n
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n
        self.pieces[int(self.n / 2) - 1][int(self.n / 2)] = 1
        self.pieces[int(self.n / 2)][int(self.n / 2) - 1] = 1
        self.pieces[int(self.n / 2) - 1][int(self.n / 2) - 1] = -1;
        self.pieces[int(self.n / 2)][int(self.n / 2)] = -1;

    def __getitem__(self, index):
        """
        Дает возможность использовать индексацию [][].
        """
        return self.pieces[index]

    def countDiff(self, color):
        """
        Подсчитывает разницу между количеством фигур цвета color и -color
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                if self[x][y] == -color:
                    count -= 1
        return count

    def getCount(self, color):
        """
        Подсчитывает количество фигур на доске заданного цвета color.
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
        return count

    def get_legal_moves(self, color):
        """
        Вычисление возможных ходов для цвета color.
        """
        moves = set()

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    newmoves = self.get_moves_for_square((x, y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        """
        Определяем, есть ли возможные ходы для цвета color.
        """
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    newmoves = self.get_moves_for_square((x, y))
                    if len(newmoves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """
        Вычисление допустимых ходов для конкретного квадрата square.
        """
        (x, y) = square
        color = self[x][y]
        if color == 0:
            return None
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                moves.append(move)
        return moves

    def execute_move(self, move, color):
        """
        Выполняет данный ход move. Переворачивает фигуры, если это необходимо.
        """
        flips = [flip for direction in self.__directions
                 for flip in self._get_flips(move, direction, color)]
        assert len(list(flips)) > 0
        for x, y in flips:
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """
        Возвращает конечную точку для допустимого хода, начинающегося в origin.
        Ходим с помощью _increment_move()
        """
        x, y = origin
        color = self[x][y]
        flips = []
        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    return x, y
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """
        Возвращает список фигур, которые нужно передвинуть для хода origin в направлении direction для
        цвета color.
        """
        flips = [origin]
        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                return flips
        return []

    @staticmethod
    def _increment_move(move, direction, n):
        """
        Генерируем ход.
        """
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < n, move)):
            yield move
            move = list(map(sum, zip(move, direction)))
