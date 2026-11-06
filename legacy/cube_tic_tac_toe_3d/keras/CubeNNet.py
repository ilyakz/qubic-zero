import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import *

class CubeNNet():
    def __init__(self, game, args):
        # game params
        self.board_z, self.board_y, self.board_x = game.getBoardSize()
        self.image_stack_size = 13
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        #self.build()
        self.build_simple()
    
    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        mc = self.args
        
        #input_boards = Input(shape=(self.board_z, self.board_y, self.board_x))    # s: batch_size x board_x x board_y
        #in_x = x = Reshape((1, self.board_z, self.board_y, self.board_x))(input_boards)
        
        in_x = x = Input((1, self.board_z, self.board_x, self.board_y))
        
        # (batch, channels, height, width)
        x = Conv3D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)
        
        for i in range(mc.residual_block_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x
        
        # for policy output
        x = Conv3D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.action_size, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="pi")(x)
       
        # for value output
        x = Conv3D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="v")(x)
        
        self.model = Model(in_x, [policy_out, value_out], name="cube_model")
        
        """
        Compiles the model to use optimizer and loss function tuned for supervised learning
        """
        
        # opt = Adam(mc.lr)
        opt = SGD(lr=mc.lr, momentum=0.9)
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        self.model.compile(optimizer=opt, loss=losses) #, loss_weights=mc.trainer_loss_weights)



    def _build_residual_block(self, x, index):
        mc = self.args
        in_x = x
        res_name = "res"+str(index)
        x = Conv3D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv3D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x


    def build_simple(self):
        """ Simple version of the NN """
        args = self.args
        self.input_boards = Input(shape=(self.board_z, self.board_y, self.board_x))    # s: batch_size x board_x x board_y

        #x_image = self.input_boards
        x_image = Reshape((self.board_z, self.board_y, self.board_x, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=4)(Conv3D(args.num_channels, 3, padding='same',use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=4)(Conv3D(args.num_channels, 3, padding='same',use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=4)(Conv3D(args.num_channels, 3, padding='same',use_bias=False)(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=4)(Conv3D(args.num_channels, 3, padding='valid',use_bias=False)(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
            
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        
    def destroy(self):
        K.clear_session()
