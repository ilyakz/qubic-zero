import sys
import tensorflow as tf
sys.path.append('..')


class OthelloNNet:
    def __init__(self, game, args):
        # Параметры игры
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Ренейм функций 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Нейронная сеть
        self.graph = tf.Graph()
        with self.graph.as_default():
            # s: batch_size x board_x x board_y
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])   
            self.dropout = tf.placeholder(tf.float32)

            # Входной слой
            # batch_size  x board_x x board_y x 1
            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])

            # Сверточный слой №1. Функция активации ReLU, также между сверточними слоями применяется Batch Normalization
            # для ускорения обучения сети
            # batch_size  x board_x x board_y x num_channels
            h_conv1 = Relu(BatchNormalization(self.conv2d(x_image, args.num_channels, 'same'), axis=3))

            # Сверточный слой №2. Функция активации ReLU, также между сверточними слоями применяется Batch Normalization
            # для ускорения обучения сети
            # batch_size  x board_x x board_y x num_channels
            h_conv2 = Relu(BatchNormalization(self.conv2d(h_conv1, args.num_channels, 'same'), axis=3))

            # Сверточный слой №3. Функция активации ReLU, также между сверточними слоями применяется Batch Normalization
            # для ускорения обучения сети
            # batch_size  x (board_x-2) x (board_y-2) x num_channels
            h_conv3 = Relu(BatchNormalization(self.conv2d(h_conv2, args.num_channels, 'valid'), axis=3))

            # Сверточный слой №4. Функция активации ReLU, также между сверточними слоями применяется Batch Normalization
            # для ускорения обучения сети
            # batch_size  x (board_x-4) x (board_y-4) x num_channels
            h_conv4 = Relu(BatchNormalization(self.conv2d(h_conv3, args.num_channels, 'valid'), axis=3))    
            h_conv4_flat = tf.reshape(h_conv4, [-1, args.num_channels*(self.board_x-4)*(self.board_y-4)])

            # Полносвязный слой с активационной функцией ReLU c применением Batch Normalization
            # batch_size x 1024
            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1)), rate=self.dropout)

            # Полносвязный слой с активационной функцией ReLU c применением Batch Normalization
            # batch_size x 512
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1)), rate=self.dropout)

            # Выходной слой Pi
            # batch_size x self.action_size
            self.pi = Dense(s_fc2, self.action_size)

            # Выходной слой V
            # batch_size x 1
            self.v = Tanh(Dense(s_fc2, 1))                                                               

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
        """
        Обертка над функцией Tensorflow для создание 2-мерного сверточного слоя
        """
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding)

    def calculate_loss(self):
        """
        Вычисление ошибки сети
        """
        # ожидаемый Pi
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        # ожидаемый V
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        # ошибка Pi
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        # ошибка V
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        # Для оптимизации сети используется Adam Optimizer
        self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)



