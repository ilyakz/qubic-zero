from framework.Arena import Arena


class TicTacToeArena(Arena):
    """
    Arena класс для крестиков - ноликов. Потомок класса Arena.
    """
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Turn ", str(it), "Player ", str(curPlayer))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        self.game.players = (1, -1)
        for _ in range(num):
            game = self.playGame(verbose=verbose)
            if game == 1:
                self.game.score[0] += 1
                oneWon += 1
            elif game == 1e-4:
                self.game.score[0] += 0
                oneWon += 0
            else:
                self.game.score[1] += 1
                twoWon += 1
        self.player1, self.player2 = self.player2, self.player1
        self.game.players = (-1, 1)
        for _ in range(num):
            game = self.playGame(verbose=verbose)
            if game == -1:
                self.game.score[0] += 1
                oneWon += 1
            elif game == 1e-4:
                self.game.score[0] += 0
                oneWon += 0
            else:
                twoWon += 1
                self.game.score[1] += 1
        return oneWon, twoWon
