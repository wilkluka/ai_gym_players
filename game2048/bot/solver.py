from datetime import datetime
from typing import List

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from game_engine.game import all_moves, GamesHistory, GameProgress
from utils.print import table_print
import pickle


BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)

MY_ACTIVATION = 'relu'

B_FILTERS_N = 512
B_FILTERS_N_2 = 4096

S_FILTERS_N = 128
S_FILTERS_N_2 = 1024

BATCH_SIZE = 2600


FILTERS_N = S_FILTERS_N
FILTERS_N_2 = S_FILTERS_N_2


WEIGHTS_FILE_PATH = "model_weights.h5"
LEARNING_THRESHOLD = 0.999
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
        self.model = self._create_ann()
        self.moves_ohe = OneHotEncoder()
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder()
        self.board_ohe.fit(BOARD_VALUES)
        self.weights_saver = None
        self.episode_counter = 0

    def transform_moves(self, moves):
        pass

    # @staticmethod
    # def transform_boards(list_of_boards):
    #     return np.array(list_of_boards).reshape([-1, 4, 4, 1])

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    @staticmethod
    def _create_ann():
        input_boards = Input(shape=(4, 4, 16), name="input_boards")
        input_move_nrs = Input(shape=(1,), name="input_move_nrs")
        # inputs = Input(shape=(4, 4, 1))
        xa = Conv2D(filters=FILTERS_N, kernel_size=(2, 1), activation=MY_ACTIVATION)(input_boards)
        xb = Conv2D(filters=FILTERS_N, kernel_size=(1, 2), activation=MY_ACTIVATION)(input_boards)
        xaa = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xa)
        xba = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xb)
        xab = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xa)
        xbb = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xb)
        x = Concatenate()([Flatten()(xx) for xx in [xaa, xab, xba, xbb]] + [input_move_nrs])
        x = Dense(64, activation='relu')(x)
        predictions = Dense(1, activation='linear')(x)
        my_optimizer = Nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

        model = Model(inputs=(input_boards, input_move_nrs), outputs=predictions)
        model.compile(
            optimizer=my_optimizer,
            # loss='categorical_crossentropy',
            loss='mean_squared_error',
            metrics=['mae']
        )
        print(model.summary())
        return model

    def train(self, games_history: GamesHistory):
        self.episode_counter += 1
        tensorboard = TensorBoard(
            log_dir="logs/episode_{}_{}".format(self.episode_counter, datetime.now().strftime("%H:%M:%S")),
            write_graph=False
        )
        self.weights_saver = Caching()
        boards, rewards, move_nrs = games_history.get_training_data_board_reward()
        boards = self.transform_boards(boards)
        boards1 = np.rot90(boards, axes=(1, 2))
        boards2 = np.rot90(boards1, axes=(1, 2))
        boards3 = np.rot90(boards2, axes=(1, 2))
        flipped0 = np.flip(boards, axis=1)
        flipped1 = np.rot90(flipped0, axes=(1, 2))
        flipped2 = np.rot90(flipped1, axes=(1, 2))
        flipped3 = np.rot90(flipped2, axes=(1, 2))
        x_boards = np.concatenate([boards, boards1, boards2, boards3, flipped0, flipped1, flipped2, flipped3], axis=0)
        y_data = np.array(rewards*8).reshape(-1, 1)
        x_move_nrs = np.array(move_nrs*8).reshape(-1, 1)
        x_boards_train, x_boards_valid, x_movenrs_train, x_movenrs_valid, y_train, y_valid = train_test_split(x_boards, x_move_nrs, y_data, test_size=0.38, shuffle=True)
        last_min_val_loss = 1e6
        stationary_state_counter = 0
        self.weights_saver.save(self.model)
        history = self.model.fit(
            x={"input_boards": x_boards_train, "input_move_nrs": x_movenrs_train},
            y=y_train,
            validation_data=({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid}, y_valid),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            callbacks=[tensorboard]
        )
        last_min_loss = min(history.history['loss'])
        step_counter = 0
        table_print("loss accuracy(mae) val_loss val_accuracy(mae)".split(), 30)
        best_loss_indicator = "*"
        is_good_epoch = True
        train_on = ['train', 'val'][1]
        # while True:
        #     history = self.model.fit(x_train, y_train, batch_size=2600, epochs=1, verbose=1)
        #     if history.history['loss'][0] < (.95 * last_min_loss):
        #         last_min_loss = history.history['loss'][0]
        #     else:
        #         break

        while True:
            history = self.model.fit(
                x={"input_boards": x_boards_train, "input_move_nrs": x_movenrs_train},
                y=y_train,
                validation_data=({"input_boards": x_boards_valid, "input_move_nrs": x_movenrs_valid}, y_valid),
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=0,
                callbacks=[tensorboard]
            )
            current_loss = history.history['loss'][0]
            curr_mae = history.history['mean_absolute_error'][0]
            current_val_loss = history.history['val_loss'][0]
            curr_val_mae = history.history['val_mean_absolute_error'][0]
            if (last_min_loss * LEARNING_THRESHOLD) > current_loss:
                best_loss_info = best_loss_indicator
                last_min_loss = current_loss
                if train_on == 'train':
                    is_good_epoch = True
            else:
                best_loss_info = " "
                if train_on == 'train':
                    is_good_epoch = False

            if (last_min_val_loss * LEARNING_THRESHOLD) > current_val_loss:
                best_val_loss_info = best_loss_indicator
                last_min_val_loss = current_val_loss
                if train_on == 'val':
                    is_good_epoch = True
            else:
                best_val_loss_info = ' '
                if train_on == 'val':
                    is_good_epoch = False

            # generate step for visualization

            if is_good_epoch:
                stationary_state_counter = 0
                if step_counter % 2 == 0:
                    self.weights_saver.save(self.model)
                step_counter += 1
            else:
                stationary_state_counter += 1
                if stationary_state_counter > 5:
                    self.weights_saver.load(self.model)
                    break

            if train_on == 'train':
                best_loss_info = '!' + best_loss_info
            else:
                best_val_loss_info = '!' + best_val_loss_info

            # print_data = [
            #     "{}{}".format(best_loss_info, current_loss),
            #     curr_mae,
            #     "{}{}".format(best_val_loss_info, current_val_loss),
            #     curr_val_mae
            # ]
            # table_print(print_data, 30)
        self.weights_saver.dump()
        print('training over')

    def predict(self, possibilities: List[GameProgress], move_nr):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        boards = self.transform_boards(boards)
        expected_values = self.model.predict({"input_boards": boards, "input_move_nrs": np.full((4, 1), move_nr)})
        return expected_values
