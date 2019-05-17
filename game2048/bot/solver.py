from typing import List

import numpy as np
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, Adam
from sklearn.preprocessing import OneHotEncoder

from bot.custom_callbacks import TensorBoardTemplate, VerboseEarlyStopping, WeightsWriter, NextBestModelCheckpoint, \
    TensorBoardCountRuns
from bot.custom_layers import vh_concat, conv_prelu, PRELU, depthwise_vh_concat, depthwise_conv_prelu
from game_engine.game import all_moves, GamesHistory, GameProgress
from sklearn.utils import shuffle

from utils.magic_collections import MList

BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)
BIG_INT = int(1e8)
MY_ACTIVATION = 'linear'

BATCH_SIZE = 1700

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
        self.board_ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
        self.board_ohe.fit(BOARD_VALUES)
        self.early_stopper_cbk = VerboseEarlyStopping(monitor=MONITOR_VALUE, min_delta=-1, verbose=1,
                                                      mode=MONITOR_MODE, baseline=None, restore_best_weights=True)
        # self.scatter_log = TBLogger("logs/scatter")
        self.tb_cbk = TensorBoardCountRuns(log_dir_template="logs/{run}", write_graph=False)
        self.tb_warm_start_cbk = TensorBoardTemplate(log_dir_template="logs/warm_start_{}", write_graph=False)
        self.model_saver_cbk = WeightsWriter(WEIGHTS_FILE_PATH)
        self.model_chckpnt_cbk = NextBestModelCheckpoint(WEIGHTS_FILE_PATH, save_best_only=True, mode='min', verbose=0,
                                                         period=20)
        self.last_max_score = 1

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    def generate_input(self, list_of_boards, to_augment=None):
        """
        to_augment variable is a collection for arrays that are 1D and we want to enlarge it 8 times
        """
        boards = self.transform_boards(list_of_boards)
        # boards = np.array(list_of_boards)
        # boards = np.log2(1 + boards)
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
        # x_boards = x_boards.reshape([-1, 4, 4, 16])
        return x_boards, to_augment

    def _create_ann(self):

        input_boards = Input(shape=(4, 4, 16), name="input_boards")
        x = input_boards

        x1 = conv_prelu(n=128, kernel_size=(1, 2), padding='valid')(x)
        x2 = conv_prelu(n=128, kernel_size=(2, 1), padding='valid')(x)
        x12 = conv_prelu(n=1024, kernel_size=(2, 1), padding='valid')(x1)
        x21 = conv_prelu(n=1024, kernel_size=(1, 2), padding='valid')(x2)
        x11 = conv_prelu(n=1024, kernel_size=(1, 2), padding='valid')(x1)
        x22 = conv_prelu(n=1024, kernel_size=(2, 1), padding='valid')(x2)
        x = Concatenate()([Flatten()(conv) for conv in [x12, x21, x11, x22, x1, x2, input_boards]])
        x = Dense(16, activation=PRELU())(x)

        # x = conv_prelu(n=1024, padding='valid')(x)  # 3, 3
        # x = conv_prelu(n=1024, padding='valid')(x)  # 2, 2
        # x = conv_prelu(n=1024, padding='valid')(x)  # 1, 1
        # x = Flatten()(x)
        # x = Dense(64, activation=PRELU())(x)

        prediction = Dense(1, activation='linear')(x)
        model = Model(inputs=[input_boards], outputs=prediction)
        self.model = model
        print(model.summary())
        self._compile_model()

    def _compile_model(self, lr=0.002):
        """ we keep here optimizers for resetting them
         we use high learning rate because later on fit we use lr_scheduler"""
        # self.my_optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0004, amsgrad=False)

        self.model.compile(optimizer=self.my_optimizer, loss='mean_squared_error')
        # if self.episode_counter >= 0:
        # self.model.load_weights(WEIGHTS_FILE_PATH)

    def train(self, games_history: GamesHistory):

        self.episode_counter += 1

        boards, scores, discounted_scores, move_nrs = games_history.get_training_data()
        final_values = [[s[-1] for ss in s] for s in scores]

        boards = MList(boards).flatten().to_ndarray()
        discounted_scores = MList(discounted_scores).flatten().to_ndarray()
        x_boards, [y] = self.generate_input(boards, [discounted_scores])

        my_callbacks = [self.tb_cbk]
        x_boards, y = shuffle(x_boards, y)
        self.last_max_score = max(self.last_max_score, y.max())
        self.model.fit(x={"input_boards": x_boards}, y=y, epochs=1, validation_split=1-GOLDEN_RATIO,
                       shuffle=True, batch_size=BATCH_SIZE, verbose=0, callbacks=my_callbacks)
        if self.episode_counter % 10 == 1:
            self.model.save_weights(WEIGHTS_FILE_PATH, overwrite=True)
            print('weights saved')
        print('training over')

    def predict(self, possibilities: List[GameProgress], move_nr):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        for b in boards:
            b.add_random2()
        x_boards, _ = self.generate_input(boards)
        expected_values_with_rotations = self.model.predict({"input_boards": x_boards})
        for b in boards:
            b.remove_random2()
        expected_values = expected_values_with_rotations.reshape(-1, len(possibilities)).mean(axis=0)
        return expected_values

    def write_errors(self, tag, x_boards_valid, x_movenrs_valid, y_valid):
        pass
        # y_pred = self.model.predict({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid},
        #                             batch_size=BATCH_SIZE)
        # self.scatter_log.log_scatter_plot(tag, y_valid, y_valid - y_pred, self.episode_counter)

    def compute_learning_threshold(self):
        return 1 - 1 / (10 * self.episode_counter)
