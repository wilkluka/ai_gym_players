from typing import List

import numpy as np
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Nadam, Adam
from sklearn.preprocessing import OneHotEncoder

from bot.custom_callbacks import TensorBoardTemplate, VerboseEarlyStopping, WeightsWriter, NextBestModelCheckpoint
from bot.custom_layers import vh_concat, conv_prelu, PRELU
from game_engine.game import all_moves, GamesHistory, GameProgress

from utils.magic_collections import MList

BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)
BIG_INT = int(1e8)
MY_ACTIVATION = 'linear'

S_FILTERS_N = 128
S_FILTERS_N_2 = 1024

BATCH_SIZE = 5500

FILTERS_N = S_FILTERS_N
FILTERS_N_2 = S_FILTERS_N_2

WEIGHTS_FILE_PATH = "model_weights.h5"
WEIGHTS_PICKLE = "weights.pickle"

GOLDEN_RATIO = 0.62
# MONITOR_VALUE = 'val_loss'
MONITOR_VALUE = 'loss'
MONITOR_MODE = 'min'


class BoardSolver:

    def __init__(self):
        self.episode_counter = 0
        self.model = None
        self._create_ann()
        self.moves_ohe = OneHotEncoder(categories='auto')
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder(categories='auto')
        self.board_ohe.fit(BOARD_VALUES)
        self.early_stopper_cbk = VerboseEarlyStopping(monitor=MONITOR_VALUE, min_delta=-1, verbose=1,
                                                      mode=MONITOR_MODE, baseline=None, restore_best_weights=True)
        # self.scatter_log = TBLogger("logs/scatter")
        self.tb_cbk = TensorBoardTemplate(log_dir_template="logs/ep_{}_patience_{}_mindelta_{:.3g}",
                                          write_graph=False, log_lr=True)
        self.tb_warm_start_cbk = TensorBoardTemplate(log_dir_template="logs/warm_start_{}", write_graph=False)
        self.model_saver_cbk = WeightsWriter(WEIGHTS_FILE_PATH)
        self.model_chckpnt_cbk = NextBestModelCheckpoint(WEIGHTS_FILE_PATH, save_best_only=True, mode='min', verbose=0,
                                                         period=20)

    # def transform_moves(self, moves):
    #     pass

    # @staticmethod
    # def transform_boards(list_of_boards):
    #     return np.array(list_of_boards).reshape([-1, 4, 4, 1])

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    def generate_input(self, list_of_boards, to_augment=None):
        """
        to_augment variable is a collection for arrays that are 1D and we want to enlarge it 8 times
        """
        boards = self.transform_boards(list_of_boards)
        boards1 = np.rot90(boards, axes=(1, 2))
        boards2 = np.rot90(boards1, axes=(1, 2))
        boards3 = np.rot90(boards2, axes=(1, 2))
        flipped0 = np.flip(boards, axis=1)
        flipped1 = np.rot90(flipped0, axes=(1, 2))
        flipped2 = np.rot90(flipped1, axes=(1, 2))
        flipped3 = np.rot90(flipped2, axes=(1, 2))
        x_boards = np.concatenate([boards, boards1, boards2, boards3, flipped0, flipped1, flipped2, flipped3], axis=0)
        if to_augment:
            to_augment = [np.tile(collection, 8).reshape(-1, 1) for collection in to_augment]
        return x_boards, to_augment

    def _create_ann(self):

        input_boards = Input(shape=(4, 4, 16), name="input_boards")
        x = vh_concat(padding='same')(input_boards)
        x = vh_concat(padding='same')(x)
        x = vh_concat(padding='same')(x)
        x = conv_prelu()(x)  # 3, 3
        x = conv_prelu()(x)  # 2, 2
        x = conv_prelu()(x)  # 1, 1
        x = Flatten()(x)
        x = Dense(64, activation=PRELU())(x)
        x = Dense(16, activation=PRELU())(x)
        prediction = Dense(1, activation='linear')(x)
        model = Model(inputs=input_boards, outputs=prediction)
        self.model = model
        print(model.summary())
        self._compile_model()

    def _compile_model(self, lr=0.002):
        # we keep here optimizers for resetting them
        # we use high learning rate because later on fit we use lr_scheduler
        # self.my_optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.my_optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0004, amsgrad=False)
        self.model.compile(optimizer=self.my_optimizer, loss='mean_squared_error')
        if self.episode_counter > 0:
            self.model.load_weights(WEIGHTS_FILE_PATH)

    def train(self, games_history: GamesHistory):
        # self._compile_model(0.002 * .95**self.episode_counter)
        self._compile_model(0.001)
        # self.model.optimizer.lr.va = K.variable(max(0.1 / 3 ** self.episode_counter, min_lr), name='lr')

        self.episode_counter += 1

        boards, scores = games_history.get_training_data()
        final_scores = [rew[-1] for rew in scores]
        max_score = max(final_scores)

        most_moves = max([len(game) for game in scores])
        patience_in_epochs = int(np.sqrt(max_score))

        self.early_stopper_cbk.patience = patience_in_epochs
        meaningful_delta = min(-2e4/(3.16 ** self.episode_counter), -1e-2)
        self.early_stopper_cbk.min_delta = meaningful_delta  # because we want loss to decrease...
        self.tb_cbk.update_log_dir(self.episode_counter, patience_in_epochs, -meaningful_delta)
        self.tb_warm_start_cbk.update_log_dir(self.episode_counter)

        val_indexes = np.array(final_scores).argsort()[-2::-3]  # indexes of games for validation
        # start from -2 and go by -3 ex. [2:] == [2::1] | [start:stop:step]

        boards_train, boards_val = MList(boards).split(val_indexes).map(lambda x: MList(x).
                                                                        flatten().to_ndarray()).to_list()
        rewards_train, rewards_val = MList(scores).split(val_indexes).map(lambda x: MList(x).
                                                                          flatten().to_ndarray()).to_list()
        x_train, [y_train] = self.generate_input(boards_train, [rewards_train])
        x_val, [y_val] = self.generate_input(boards_val, [rewards_val])

        my_callbacks = [self.tb_cbk, self.early_stopper_cbk, self.model_chckpnt_cbk]
        # my_callbacks = [lr_reducer_cbk, self.tb_cbk, self.early_stopper_cbk, self.model_saver_cbk]
        # my_callbacks = [self.tb_cbk, self.model_chckpnt_cbk]

        print('train', x_train.shape, y_train.shape, 'val', x_val.shape, y_val.shape)

        several = 20
        print("performing warm start with {} epochs".format(several))
        # history = self.model.fit(x=x_train, y=y_train, epochs=several, validation_data=(x_val, y_val), shuffle=True,
        #                          batch_size=BATCH_SIZE, verbose=0, callbacks=[self.tb_warm_start_cbk])
        # print("val_losses", history.history['val_loss'])

        self.model.fit(x=x_train, y=y_train, epochs=BIG_INT, validation_data=(x_val, y_val), shuffle=True,
                       batch_size=BATCH_SIZE, verbose=0, callbacks=my_callbacks)

        print('training over')

    def predict(self, possibilities: List[GameProgress]):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        # here we
        x_boards, _ = self.generate_input(boards)
        expected_values_with_rotations = self.model.predict(x_boards)
        expected_values = expected_values_with_rotations.reshape(-1, len(possibilities)).mean(axis=0)
        return expected_values

    def write_errors(self, tag, x_boards_valid, x_movenrs_valid, y_valid):
        pass
        # y_pred = self.model.predict({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid},
        #                             batch_size=BATCH_SIZE)
        # self.scatter_log.log_scatter_plot(tag, y_valid, y_valid - y_pred, self.episode_counter)

    def compute_learning_threshold(self):
        return 1 - 1 / (10 * self.episode_counter)
