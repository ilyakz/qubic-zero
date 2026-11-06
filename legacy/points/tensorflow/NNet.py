from framework.utils import *
from utils import Bar, AverageMeter
from framework.NeuralNet import NeuralNet
import tensorflow as tf
from .OthelloNNet import OthelloNNet as onnet
import os
import time
import numpy as np
import sys
sys.path.append('../../')


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.sess = tf.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: Список примеров. Каждый пример представляется в форме (board, pi, v).
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Предсказываем, вычисляем градиент и выполняем шаг стохастического градиентного спуска
                input_dict = {self.nnet.input_boards: boards, self.nnet.target_pis: pis, self.nnet.target_vs: vs,
                              self.nnet.dropout: args.dropout}

                data_time.update(time.time() - end)

                # Потери
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict)
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}' \
                             ' | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.\
                    format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()

    def predict(self, board):
        """
        board: Доска, преобразованная с помощью np.array()
        """
        # Подготовка входных данных
        board = board[np.newaxis, :, :]

        # Запуск
        pi, v = self.sess.run([self.nnet.pi, self.nnet.v], feed_dict={self.nnet.input_boards: board,
                                                                      self.nnet.dropout: 0})

        pi = np.exp(pi) / np.sum(np.exp(pi))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver is None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            pass
        with self.nnet.graph.as_default():
            self.saver = tf.train.import_meta_graph(filepath + '.meta')
            self.saver.restore(self.sess, filepath)
