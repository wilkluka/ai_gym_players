import numpy as np
from bisect import bisect
from random import random, shuffle, sample
from tqdm import tqdm
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, DepthwiseConv2D, Concatenate, Multiply
from keras.models import Model

from sklearn.preprocessing import OneHotEncoder

import warnings
from collections import Counter
from game_engine.game import Move, Game, all_moves, GamesHistory

HISTORY_LIMIT = 20


# class BasicSolver:
#
#     def __init__(self):
#         self.model = self._create_ann()
#         self.ohe = None
#
#     @staticmethod
#     def _create_ann():
#         inputs = Input(shape=(16,))
#         x0 = Dense(64, activation='relu')(inputs)
#         x = Dense(64, activation='relu')(x0)
#         x = Dropout(.3)(Dense(64, activation='relu')(x))
#         x = Dropout(.3)(Dense(64, activation='relu')(x))
#         x = Dropout(.3)(Dense(64, activation='relu')(x))
#         x = Dropout(.3)(Dense(32, activation='relu')(x))
#         x = Dense(16, activation='relu')(x)
#         # x = Concatenate()([x, x0])
#         x = Dense(16, activation='relu')(x)
#         predictions = Dense(4, activation='softmax')(x)
#
#         model = Model(inputs=inputs, outputs=predictions)
#         model.compile(
#             optimizer='adam',
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )
#         print(model.summary())
#         return model
#
#     def train(self, history):
#         history = sum(history, [])
#         moves, boards = map(list, zip(*history))
#         boards = np.array([board.reshape([1, 16])[0] for board in boards])
#         self.ohe = OneHotEncoder()
#         true_moves = self.ohe.fit_transform([[m.value] for m in moves])
#         self.model.fit(boards, true_moves, epochs=10, verbose=1)
#
#     def predict_move(self, board):
#         board = board.reshape([1, 16])
#         return [Move(x) for x in (np.argsort(self.model.predict(board))[0] + 1)][::-1]

board_values = np.array([0] + [2**i for i in range(1, 16)]).reshape(-1, 1)

my_activation = 'tanh'

FILTERS_N = 128
FILTERS_N_2 = 1024

B_FILTERS_N = 512
B_FILTERS_N_2 = 4096


class BoardSolver:

    def __init__(self):
        self.model = self._create_ann()
        self.moves_ohe = OneHotEncoder()
        self.moves_ohe.fit([[m.value] for m in all_moves])
        self.board_ohe = OneHotEncoder()
        self.board_ohe.fit(board_values)

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
        xa = Conv2D(filters=128, kernel_size=(2, 1), activation=my_activation)(inputs)
        xb = Conv2D(filters=128, kernel_size=(1, 2), activation=my_activation)(inputs)
        xaa = Conv2D(filters=1024, kernel_size=(2, 1), activation=my_activation)(xa)
        xba = Conv2D(filters=1024, kernel_size=(2, 1), activation=my_activation)(xb)
        xab = Conv2D(filters=1024, kernel_size=(1, 2), activation=my_activation)(xa)
        xbb = Conv2D(filters=1024, kernel_size=(1, 2), activation=my_activation)(xb)
        x = Concatenate()([Flatten()(xx) for xx in [xaa, xab, xba, xbb]])
        x = Dense(64, activation=my_activation)(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            # loss='mean_squared_error',
            metrics=['accuracy']
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
        moves, boards = games_history.get_training_data()
        boards = self.transform_boards(boards)
        moves_count = Counter(moves)
        print(moves_count)
        true_moves = self.moves_ohe.transform([[m.value] for m in moves])
        last_min_loss = 1e6
        stationary_state_counter = 0
        self.model.fit(boards, true_moves, batch_size=3000, epochs=2, verbose=0)
        while True:
            history = self.model.fit(boards, true_moves, batch_size=3000, epochs=1, verbose=0, validation_split=.2)
            current_min_loss = min(history.history['val_loss'])
            print('loss:\t\t{}\taccuracy:\t{}'.format(min(history.history['loss']), min(history.history['acc'])))
            print('val loss:\t\t{}\tval accuracy:\t\t{}'.format(min(history.history['val_loss']), min(history.history['val_acc'])))
            if current_min_loss < 4e-4:
                self.model.save_weights("model_weights.h5")
                break
            if (last_min_loss * 0.997) > current_min_loss:
                last_min_loss = current_min_loss
                stationary_state_counter = 0
                self.model.save_weights("model_weights.h5")
            else:
                stationary_state_counter += 1
                if stationary_state_counter > 3:
                    self.model.load_weights("model_weights.h5")
                    break

    # def predict_move(self, board, _reshaped_board=np.zeros([1, 4, 4, 1])):
    #     np.copyto(_reshaped_board[0, :, :, 0], board)
    #     return [Move(x) for x in (np.argsort(self.model.predict(_reshaped_board))[0] + 1)][::-1]

    def predict_move(self, board):
        reshaped_board = self.transform_boards([board])
        return [Move(x) for x in (np.argsort(self.model.predict(reshaped_board))[0] + 1)][::-1]


