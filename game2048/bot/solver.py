from typing import List

import numpy as np
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from game_engine.game import all_moves, GamesHistory, GameProgress
import pickle

from utils.magic_collections import MagicList
from utils.tb import TBLogger

BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)
BIG_INT = int(1e8)
MY_ACTIVATION = 'linear'

S_FILTERS_N = 128
S_FILTERS_N_2 = 1024

BATCH_SIZE = 1000

FILTERS_N = S_FILTERS_N
FILTERS_N_2 = S_FILTERS_N_2

WEIGHTS_FILE_PATH = "model_weights.h5"
WEIGHTS_PICKLE = "weights.pickle"

GOLDEN_RATIO = 0.62


class WeightsWriter(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.file_path, overwrite=True)


class BoardSolver:

    def __init__(self):
        self.model = None
        self._create_ann()
        self.moves_ohe = OneHotEncoder(categories='auto')
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder(categories='auto')
        self.board_ohe.fit(BOARD_VALUES)
        self.early_stopper_cbk = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=0, mode='min',
                                               baseline=None, restore_best_weights=True)

        self.episode_counter = 0
        self.scatter_log = TBLogger("logs/scatter")
        self.tb_cbk = TensorBoard(log_dir="logs/episode_{}".format(self.episode_counter), write_graph=False)
        self.model_saver_cbk = WeightsWriter(WEIGHTS_FILE_PATH)

    def transform_moves(self, moves):
        pass

    # @staticmethod
    # def transform_boards(list_of_boards):
    #     return np.array(list_of_boards).reshape([-1, 4, 4, 1])

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    def generate_input(self, list_of_boards, to_augment=None):
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
            to_augment = [np.array(collection*8).reshape(-1, 1) for collection in to_augment]
        return x_boards, to_augment

    def _create_ann(self):
        input_boards = Input(shape=(4, 4, 16), name="input_boards")
        xa = Conv2D(filters=FILTERS_N, kernel_size=(2, 1), activation=MY_ACTIVATION)(input_boards)
        xb = Conv2D(filters=FILTERS_N, kernel_size=(1, 2), activation=MY_ACTIVATION)(input_boards)
        x = Concatenate()([Flatten()(xx) for xx in [xa, xb]])
        x1 = Dense(64, activation='linear')(x)
        x2 = Dense(64, activation='exponential')(x)
        x = Concatenate()([x1, x2])
        predictions = Dense(1, activation='linear')(x)
        # my_optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

        model = Model(inputs=input_boards, outputs=predictions)
        self.model = model
        self._compile_model()
        print(model.summary())

    def _compile_model(self):
        self.model.compile(
            optimizer=self.my_optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )

    def train(self, games_history: GamesHistory):
        self.episode_counter += 1

        boards, rewards = games_history.get_training_data()

        most_moves = max([len(game) for game in rewards])
        patience_in_epochs = np.sqrt(most_moves) * 2
        self.early_stopper_cbk.patience = patience_in_epochs
        self.tb_cbk.log_dir = "logs/episode_{}_patience_{}".format(self.episode_counter, patience_in_epochs)
        # split games at random
        train_test_data = train_test_split(boards, rewards, train_size=GOLDEN_RATIO, shuffle=True)
        boards_train, boards_val, rewards_train, rewards_val = [MagicList(data).flatten() for data in train_test_data]

        x_train, [y_train] = self.generate_input(boards_train, [rewards_train])
        x_val, [y_val] = self.generate_input(boards_val, [rewards_val])

        my_callbacks = [self.tb_cbk, self.early_stopper_cbk, self.model_saver_cbk]

        self.model.fit(x=x_train, y=y_train, epochs=BIG_INT, validation_data=(x_val, y_val),
                       batch_size=BATCH_SIZE, verbose=0, callbacks=my_callbacks)

        print('training over')

    def predict(self, possibilities: List[GameProgress]):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        x_boards, _ = self.generate_input(boards)
        expected_values_with_rotations = self.model.predict(x_boards)
        expected_values = expected_values_with_rotations.reshape(-1, len(possibilities)).mean(axis=0)
        # expected_values = np.exp(expected_values)-800
        return expected_values

    def write_errors(self, tag, x_boards_valid, x_movenrs_valid, y_valid):
        pass
        # y_pred = self.model.predict({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid},
        #                             batch_size=BATCH_SIZE)
        # self.scatter_log.log_scatter_plot(tag, y_valid, y_valid - y_pred, self.episode_counter)

    def compute_learning_threshold(self):
        return 1 - 1 / (10 * self.episode_counter)
