from typing import List

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from game_engine.game import all_moves, GamesHistory, GameProgress
import pickle

from utils.tb import TBLogger

BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)

MY_ACTIVATION = 'linear'

S_FILTERS_N = 128
S_FILTERS_N_2 = 1024

BATCH_SIZE = 1000

FILTERS_N = S_FILTERS_N
FILTERS_N_2 = S_FILTERS_N_2

WEIGHTS_FILE_PATH = "model_weights.h5"
WEIGHTS_PICKLE = "weights.pickle"


class WeightsSaver:

    def save(self, model):
        raise NotImplementedError

    def load(self, model):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class Caching(WeightsSaver):

    def __init__(self):
        self.weights = None

    def save(self, model):
        print("weights saved")
        self.weights = model.get_weights()

    def load(self, model):
        if self.weights:
            print("weights loaded")
            model.set_weights(self.weights)

    def dump(self):
        if self.weights:
            with open(WEIGHTS_PICKLE, "wb") as ffile:
                pickle.dump(self.weights, ffile)
            print("weights dumped to {}".format(WEIGHTS_PICKLE))


class Saver(WeightsSaver):

    def save(self, model):
        model.save_weights(WEIGHTS_FILE_PATH)

    def load(self, model):
        model.load_weights(WEIGHTS_FILE_PATH)

    def dump(self):  # nothing to do
        pass


class BoardSolver:

    def __init__(self):
        self.model = None
        self._create_ann()
        self.moves_ohe = OneHotEncoder()
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder()
        self.board_ohe.fit(BOARD_VALUES)
        self.weights_saver = Caching()
        self.episode_counter = 0
        self.scatter_log = TBLogger("logs/scatter")
        self.tb = TensorBoard(
            log_dir="logs/episode_{}".format(self.episode_counter),
            write_graph=True
        )

    def transform_moves(self, moves):
        pass

    # @staticmethod
    # def transform_boards(list_of_boards):
    #     return np.array(list_of_boards).reshape([-1, 4, 4, 1])

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    def generate_rotated_boards(self, list_of_boards):
        boards = self.transform_boards(list_of_boards)
        boards1 = np.rot90(boards, axes=(1, 2))
        boards2 = np.rot90(boards1, axes=(1, 2))
        boards3 = np.rot90(boards2, axes=(1, 2))
        flipped0 = np.flip(boards, axis=1)
        flipped1 = np.rot90(flipped0, axes=(1, 2))
        flipped2 = np.rot90(flipped1, axes=(1, 2))
        flipped3 = np.rot90(flipped2, axes=(1, 2))
        x_boards = np.concatenate([boards, boards1, boards2, boards3, flipped0, flipped1, flipped2, flipped3], axis=0)
        return x_boards

    def _create_ann(self):
        input_boards = Input(shape=(4, 4, 16), name="input_boards")
        input_move_nrs = Input(shape=(1,), name="input_move_nrs")
        xa = Conv2D(filters=FILTERS_N, kernel_size=(2, 1), activation=MY_ACTIVATION)(input_boards)
        xb = Conv2D(filters=FILTERS_N, kernel_size=(1, 2), activation=MY_ACTIVATION)(input_boards)
        xaa = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xa)
        xba = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xb)
        xab = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xa)
        xbb = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xb)
        x = Concatenate()([Flatten()(xx) for xx in [xaa, xab, xba, xbb, xa, xb]] + [input_move_nrs])
        x1 = Dense(64, activation='linear')(x)
        x2 = Dense(64, activation='exponential')(x)
        x = Concatenate()([x1, x2])
        predictions = Dense(1, activation='linear')(x)
        # my_optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

        model = Model(inputs=(input_boards, input_move_nrs), outputs=predictions)
        self.model = model
        self._compile_model()
        print(model.summary())

    def _compile_model(self):
        self.model.compile(
            optimizer=self.my_optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'logcosh']
        )

    def train(self, games_history: GamesHistory):
        self.episode_counter += 1
        self.tb = TensorBoard(
            log_dir="logs/episode_{}".format(self.episode_counter),
            write_graph=False
        )
        learning_threshold = self.compute_learning_threshold()
        boards, rewards, move_nrs = games_history.get_training_data_board_reward()
        x_boards = self.generate_rotated_boards(boards)

        y_data = np.array(rewards*8).reshape(-1, 1)
        # y_data = np.log(800 + y_data)
        x_move_nrs = np.array(move_nrs*8).reshape(-1, 1)
        train_test_dataset = train_test_split(x_boards, x_move_nrs, y_data, test_size=0.38, shuffle=True, random_state=self.episode_counter)
        x_boards_train, x_boards_valid, x_movenrs_train, x_movenrs_valid, y_train, y_valid = train_test_dataset
        stationary_state_counter = 0
        self.write_errors("errors/pre_train", x_boards_valid, x_movenrs_valid, y_valid)
        self.weights_saver.save(self.model)
        self._compile_model()
        self.weights_saver.load(self.model)
        history = self.model.fit(
            x={"input_boards": x_boards_train, "input_move_nrs": x_movenrs_train},
            y=y_train,
            validation_data=({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid}, y_valid),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            callbacks=[self.tb]
        )
        step_counter = 0
        train_on = ['train', 'val'][0]
        is_good_epoch = True

        last_min_loss = history.history['loss'][0]
        last_min_val_loss = history.history['val_loss'][0]
        while True:
            history = self.model.fit(
                x={"input_boards": x_boards_train, "input_move_nrs": x_movenrs_train},
                y=y_train,
                validation_data=({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid}, y_valid),
                batch_size=BATCH_SIZE,
                epochs=step_counter+1,
                verbose=0,
                callbacks=[self.tb],
                initial_epoch=step_counter
            )
            current_loss = history.history['loss'][0]
            current_val_loss = history.history['val_loss'][0]
            if (last_min_loss * learning_threshold) > current_loss:
                last_min_loss = current_loss
                if train_on == 'train':
                    is_good_epoch = True
            else:
                if train_on == 'train':
                    is_good_epoch = False

            if (last_min_val_loss * learning_threshold) > current_val_loss:
                last_min_val_loss = current_val_loss
                if train_on == 'val':
                    is_good_epoch = True
            else:
                if train_on == 'val':
                    is_good_epoch = False

            if is_good_epoch:
                stationary_state_counter = 0
                self.weights_saver.save(self.model)
                step_counter += 1
            else:
                stationary_state_counter += 1
                if stationary_state_counter > 30:
                    self.weights_saver.load(self.model)
                    break

        self.write_errors("errors/post_train", x_boards_valid, x_movenrs_valid, y_valid)
        self.weights_saver.dump()
        print('training over')

    def predict(self, possibilities: List[GameProgress], move_nr):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        x_boards = self.generate_rotated_boards(boards)
        expected_values_with_rotations = self.model.predict({"input_boards": x_boards, "input_move_nrs": np.full((4*8, 1), move_nr)})
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
