from collections import deque
from framework.Arena import Arena
from framework.MCTS import MCTS
import numpy as np
import utils as ut
import time


class Coach:
    """
    Данный класс описывает процесс игры нейронной сети с самой собой и ее обучение.
    Он использует функции, определенные в классах Game и NeuralNet. Параметр args
    уточняется в main.py
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def executeEpisode(self):
        """
        Данный метод выполняет один эпизод игры нейронной сети с самой собой
        от лица первого игрока. Всякая сыгранная игра добавляется в тренировочный
        дата - сет trainExamples. Игра играется до тех пор, пока не закончится.
        После окончания игры, ее исход используется для присваивания значений
        каждому примеру в trainExamples.
        Используется temp = 1, если episodeStep < tempThreshold, а затем temp = 0.
        Returns:
            trainExamples:
                Список примеров, каждый из которых представляется в виде (canonicalBoard, pi, v),
                где pi - вектор вероятностей (полтики) для метода Монте - Карло. v = 1, если игрок
                выиграл, а иначе v = -1.
        """
        trainExamples = []
        self.board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(self.board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

            r = self.game.getGameEnded(self.board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Выполняет количество итераций, равное numIters с числом эпизодов, равным numEps, игр нейронной сети
        с самой собой. После каждой итерации, происходит перетренировка нейронной сети на примерах из
        trainExamples. Далее новая нейронная сеть соревнуется со старой. Новая нейронная сеть принимается
        как "хорошая", если нормализованное количество ее побед >= updateThreshold.
        """

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for i in range(self.args.numIters):
            # bookkeeping
            print('------ITER ' + str(i+1) + '------')
            eps_time = ut.AverageMeter()
            bar = ut.Bar('Self Play', max=self.args.numEps)
            end = time.time()

            for eps in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                trainExamples += self.executeEpisode()                

                # bookkeeping + plot progress
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.\
                    format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pnet = self.nnet.__class__(self.game)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)
            # print("Loss_pi:{lp}, Loss_v:{lv}".format(lp=self.nnet.loss_pi, lv=self.nnet.loss_v))
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : ' + str(nwins) + '/' + str(pwins))
            if pwins + nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet = pnet

            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i) + '.pth.tar')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
