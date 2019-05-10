class Arena:
    """
    Класс, описывающий арену, в которой любые два агента могут сразиться друг с другом.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1, 2: Две функции, принимающие на вход состояние доски и возвращающие действие.
            game: Объект класса Game.
            display:
                Функция, принимающая на вход состояние доски и печатающая ее.
                Необходима для verbose - мода.
                Примеры:
                    othello/OthelloGame
                    tic_tac_toe/TicTacToeGame

        Примеры агентов можно посмотреть в
            othello/OthelloPlayers.py
            tic_tac_toe/TicTacToePlayers.py

        В модуле pit.py можно столкнуть различных агентов.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Играет одну игру.

        Returns:
            winner: Игрок, победивший в игре (1 если player1, -1 если player2).
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.getInitBoard()
        it = 0

        while self.game.getGameEnded(board, cur_player) == 0:
            it += 1

            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(cur_player))
                self.display(board)

            action = players[cur_player+1](self.game.getCanonicalForm(board, cur_player))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, cur_player), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0

            board, cur_player = self.game.getNextState(board, cur_player, action)

        if verbose:
            assert self.display
            print("Turn ", str(it), "Player ", str(cur_player))
            self.display(board)

        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Играет количество игр, равное num, в которых игрок player1 начинает num/2 игр, а затем
        игрок player2 начинает оставшееся количество игр.

        Returns:
            oneWon: Количество игр, выигранных player1.
            twoWon: Количество игр, выигранных player2.
        """

        num = int(num/2)
        one_won = 0
        two_won = 0

        for _ in range(num):
            if self.playGame(verbose=verbose) == 1:
                one_won += 1
            else:
                two_won += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            if self.playGame(verbose=verbose) == -1:
                one_won += 1
            else:
                two_won += 1

        return one_won, two_won
