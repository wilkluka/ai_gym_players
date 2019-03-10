from typing import List

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.preprocessing import OneHotEncoder

from game_engine.game import all_moves, GamesHistory, GameProgress


BOARD_VALUES = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)

MY_ACTIVATION = 'relu'

FILTERS_N = 128
FILTERS_N_2 = 1024

B_FILTERS_N = 512
B_FILTERS_N_2 = 4096

WEIGHTS_FILE_PATH = "model_weights.h5"


class BoardSolver:

    def __init__(self):
        self.model = self._create_ann()
        self.moves_ohe = OneHotEncoder()
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder()
        self.board_ohe.fit(BOARD_VALUES)

    def transform_moves(self, moves):
        pass

    # def transform_boards(list_of_boards):
    #     return np.array(list_of_boards).reshape([-1, 4, 4, 1])

    def transform_boards(self, list_of_boards):
        array_of_boards = np.array(list_of_boards)
        one_hot_sparse = self.board_ohe.transform(array_of_boards.reshape([-1, 1]))
        return one_hot_sparse.toarray().reshape([-1, 4, 4, 16])

    @staticmethod
    def _create_ann():
        inputs = Input(shape=(4, 4, 16))
        xa = Conv2D(filters=FILTERS_N, kernel_size=(2, 1), activation=MY_ACTIVATION)(inputs)
        xb = Conv2D(filters=FILTERS_N, kernel_size=(1, 2), activation=MY_ACTIVATION)(inputs)
        xaa = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xa)
        xba = Conv2D(filters=FILTERS_N_2, kernel_size=(2, 1), activation=MY_ACTIVATION)(xb)
        xab = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xa)
        xbb = Conv2D(filters=FILTERS_N_2, kernel_size=(1, 2), activation=MY_ACTIVATION)(xb)
        x = Concatenate()([Flatten()(xx) for xx in [xaa, xab, xba, xbb]])
        x = Dense(64, activation='relu')(x)
        predictions = Dense(1, activation='linear')(x)
        nadam_custom = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(
            optimizer=nadam_custom,
            # loss='categorical_crossentropy',
            loss='mean_squared_error',
            metrics=['mae']
        )
        print(model.summary())
        return model
    #
    # def train(self, games_history: GamesHistory):
    #     moves, boards = games_history.get_training_data()
    #     boards = self.transform_boards(boards)
    #     moves_count = Counter(moves)
    #     print(moves_count)
    #     true_moves = self.moves_ohe.transform([[m.value] for m in moves])
    #     last_min_loss = 1e6
    #     stationary_state_counter = 0
    #     self.model.fit(boards, true_moves, batch_size=3000, epochs=2, verbose=0)
    #     while True:
    #         history = self.model.fit(boards, true_moves, batch_size=3000, epochs=1, verbose=0, validation_split=.36)
    #         current_min_loss = min(history.history['val_loss'])
    #         print('loss:\t\t{}\taccuracy:\t{}'.format(min(history.history['loss']), min(history.history['acc'])))
    #         print('val loss:\t\t{}\tval accuracy:\t\t{}'.format(min(history.history['val_loss']), min(history.history['val_acc'])))
    #         if (last_min_loss * 0.99997) > current_min_loss:
    #             last_min_loss = current_min_loss
    #             stationary_state_counter = 0
    #             self.model.save_weights("model_weights.h5")
    #         else:
    #             stationary_state_counter += 1
    #             if stationary_state_counter > 4:
    #                 break
    #     self.model.load_weights("model_weights.h5")

    def train(self, games_history: GamesHistory):
        boards, rewards = games_history.get_training_data_board_reward()
        boards = self.transform_boards(boards)
        boards1 = np.rot90(boards, axes=(1,2))
        boards2 = np.rot90(boards1, axes=(1, 2))
        boards3 = np.rot90(boards2, axes=(1, 2))
        flipped0 = np.flip(boards, axis=1)
        flipped1 = np.rot90(flipped0, axes=(1, 2))
        flipped2 = np.rot90(flipped1, axes=(1, 2))
        flipped3 = np.rot90(flipped2, axes=(1, 2))
        x_train = np.concatenate([boards, boards1, boards2, boards3, flipped0, flipped1, flipped2, flipped3], axis=0)
        y_train = np.array(rewards*8).reshape(-1, 1)
        last_min_loss = 1e6
        stationary_state_counter = 0
        self.model.fit(x_train, y_train, batch_size=2000, epochs=2, verbose=0)
        step_counter = 0
        while True:
            history = self.model.fit(x_train, y_train, batch_size=2000, epochs=1, verbose=0)
            current_min_loss = min(history.history['loss'])
            print('loss:\t\t{}\taccuracy:\t{}'.format(min(history.history['loss']), min(history.history['mean_absolute_error'])))
            # print('val loss:\t\t{}\tval accuracy:\t\t{}'.format(min(history.history['val_loss']), min(history.history['val_mean_absolute_error'])))
            if (last_min_loss * 0.997) > current_min_loss:
                last_min_loss = current_min_loss
                stationary_state_counter = 0
                step_counter += 1
                if step_counter % 3 == 1:
                    self.model.save_weights(WEIGHTS_FILE_PATH)
            else:
                stationary_state_counter += 1
                if stationary_state_counter > 3:
                    self.model.load_weights(WEIGHTS_FILE_PATH)
                    break
        print('training over')

    # def predict_move(self, board, _reshaped_board=np.zeros([1, 4, 4, 1])):
    #     np.copyto(_reshaped_board[0, :, :, 0], board)
    #     return [Move(x) for x in (np.argsort(self.model.predict(_reshaped_board))[0] + 1)][::-1]

    def predict(self, possibilities: List[GameProgress]):
        moves, boards = zip(*[(possibility.move, possibility.board) for possibility in possibilities])
        boards = self.transform_boards(boards)
        expected_values = self.model.predict(boards)
        return expected_values

        # possibilities
        # reshaped_board = self.transform_boards([board])
        # return [Move(x) for x in (np.argsort(self.model.predict(reshaped_board))[0] + 1)][::-1]


