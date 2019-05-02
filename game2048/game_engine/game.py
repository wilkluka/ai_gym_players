from enum import Enum
from bisect import insort
from typing import Optional

import numpy as np
from functools import total_ordering
from random import shuffle
import pickle


BOARD_SIZE = 4


class Move(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


all_moves = list(Move.__members__.values())


class GameProgress:

    def __init__(self, move, is_possible, board=None):
        self.move = move
        self.is_possible = is_possible
        self.board = board if board is not None else Board.get_empty()

    def __hash__(self):
        return self.board.value() + hash(self.move) * 100000

    def __eq__(self, other):
        return hash(self) == hash(other)


class Board(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'uint16'
        this = np.array(*args, **kwargs)
        this = np.asarray(this).view(cls)
        return this

    def __array_finalize__(self, obj):
        pass

    def value(self) -> int:
        return int(sum(v * (np.log2(max(v, 1)) - 1) for v in self.flatten()))

    @staticmethod
    def get_empty():
        return Board(np.zeros([BOARD_SIZE, BOARD_SIZE], dtype='uint16'))

    def __repr__(self):
        return "Board({})".format(self.tolist())

    def add_random2(self):
        indexes = np.where(self == 0)
        insert_point_index = np.random.choice(np.arange(0, indexes[0].size))
        self[indexes[0][insert_point_index], indexes[1][insert_point_index]] = 2
        self.random2 = (indexes[0][insert_point_index], indexes[1][insert_point_index])

    def remove_random2(self):
        self[self.random2] = 0


class GameRound:

    def __init__(self, game_round_num: int, move: Move, current_score: float, board: Optional[Board] = None):
        self.move = move
        self.game_round_num = game_round_num
        self.board = board if board is not None else Board.get_empty()
        self.score = current_score

    def print(self, minimal: bool):
        if not minimal:
            print("{}\t{}\t{}".format(self.score, self.move, self.game_round_num))
        print(self.board)


@total_ordering
class PastGame:

    def __init__(self):
        self.final_score = 0
        self.rounds = []

    def add_round(self, game_round: GameRound):
        self.rounds.append(game_round)

    def __eq__(self, other) -> bool:
        return self.final_score == other.final_score

    def __lt__(self, other) -> bool:
        return self.final_score < other.final_score

    def compute_score(self):
        self.final_score = self.rounds[-1].score
        return self.final_score

    def print(self):
        for roundd in self.rounds:
            roundd.print()


class Game:

    def __init__(self):
        self.move_count = 0
        self.board = Board.get_empty()
        self._add_new_board_tile()
        self._add_new_board_tile()
        self.possible_moves = {move: GameProgress(move, True) for move in all_moves}
        self._refresh_possible_moves()
        self.current_score = 0

    def _set_board(self, board, add_new_tile, refresh_moves) -> None:
        np.copyto(self.board, board)
        if add_new_tile:
            self._add_new_board_tile()
        if refresh_moves:
            self._refresh_possible_moves()

    def _add_new_board_tile(self) -> None:
        if np.all(self.board > 0):
            raise ValueError
        value_to_insert = np.random.choice(a=[2, 4], size=1, p=[0.9, 0.1])
        indexes = np.where(self.board==0)
        insert_point_index = np.random.choice(np.arange(0, indexes[0].size))
        self.board[indexes[0][insert_point_index], indexes[1][insert_point_index]] = value_to_insert

    def _make_move(self, move) -> None:
        if move in (Move.UP, Move.DOWN):
            orientation = 1 if move == Move.UP else -1
            for col in range(0, BOARD_SIZE):
                last = -1
                new_col = []
                for row in range(0, BOARD_SIZE)[::orientation]:
                    curr = self.board[row, col]
                    if last == curr:
                        curr += curr
                        new_col.append(curr)
                        last = -1
                    elif curr != 0:
                        if last > 0:
                            new_col.append(last)
                        last = curr
                if last > 0:
                    new_col.append(last)
                # set col
                for row in range(0, BOARD_SIZE)[::orientation]:
                    if new_col:
                        self.possible_moves[move].board[row, col] = new_col[0]
                        new_col.pop(0)
                    else:
                        self.possible_moves[move].board[row, col] = 0
        else:  # move in (Moves.LEFT, Moves.RIGHT):
            orientation = 1 if move == Move.LEFT else -1
            for row in range(0, BOARD_SIZE):
                last = -1
                new_row = []
                for col in range(0, BOARD_SIZE)[::orientation]:
                    curr = self.board[row, col]
                    if last == curr:
                        curr += curr
                        new_row.append(curr)
                        last = -1
                    elif curr != 0:
                        if last > 0:
                            new_row.append(last)
                        last = curr
                if last > 0:
                    new_row.append(last)
                # set col
                for col in range(0, BOARD_SIZE)[::orientation]:
                    if new_row:
                        self.possible_moves[move].board[row, col] = new_row[0]
                        new_row.pop(0)
                    else:
                        self.possible_moves[move].board[row, col] = 0
        self.possible_moves[move].is_possible = not np.all(self.board == self.possible_moves[move].board)

    def _refresh_possible_moves(self):
        for move in all_moves:
            self._make_move(move)

    def move(self, move):
        next_move = self.possible_moves[move]
        if not next_move.is_possible:
            return False
        self.current_score += next_move.board.value() - self.board.value()
        self._set_board(next_move.board, add_new_tile=True, refresh_moves=True)
        self.move_count += 1
        return True

    def is_game_over(self):
        for move in all_moves:
            if self.possible_moves[move].is_possible:
                return False
        return True  # otherwise

    def set_from_other(self, board) -> None:
        self._set_board(board, add_new_tile=False, refresh_moves=True)

    def get_board_value(self) -> float:
        return sum(v * (np.log2(max(v, 1)) - 1) for v in self.board.flatten())


class GamesHistory:

    def __init__(self, info, limit=1000):
        self.games = []
        self.limit = limit
        self.info = info

    @staticmethod
    def merge(first, second):
        new_ob = GamesHistory("merged", limit=second.limit+first.limit)
        for game in first.games + second.games:
            new_ob.add_game(game)
        return new_ob

    def add_game(self, past_game: PastGame):
        insort(self.games, past_game)
        while len(self.games) > self.limit:
            self.games.pop(0)

    def update(self, other):
        for game in other.games:
            self.add_game(game)

    def print(self, last_n=10):
        last_n = min(last_n, self.limit)
        print(last_n, self.limit)
        print('#########################')
        print(self.info)
        for game in self.games[::-1]:
            print(game.final_score, '\t', game.rounds[-1].game_round_num, '\t', game.rounds[-1].move)
            print(game.rounds[-1].board)
            print('===========')
        print('#########################')

    def get_training_data(self, dump=True):
        all_boards = []
        all_move_nrs = []
        all_discounted_scores = []
        all_final_scores = []
        all_shifted_scores = []
        discount_scores = 0.8
        # reward_part = 0.3
        for game in self.games:
            move_nrs = [rnd.game_round_num for rnd in game.rounds]
            scores = [rnd.score for rnd in game.rounds]
            rewards = [nxt - curr for nxt, curr in zip(scores[1:], scores)]
            boards = [rnd.board for rnd in game.rounds]
            discounted_scores = []
            all_shifted_scores.append(scores[1:] + [scores[-1]])
            acc = int(boards[-1].max())
            for s in rewards[::-1]:
                acc = (acc + s) * discount_scores
                discounted_scores.append(acc)
            discounted_scores.reverse()

            all_discounted_scores.append([0] + discounted_scores)
            all_boards.append(boards)
            all_move_nrs.append(move_nrs)
            all_final_scores.append([scores[-1]] * len(scores))
            assert len(all_shifted_scores[-1]) == len(all_boards[-1])
            assert len(all_boards[-1]) == len(all_discounted_scores[-1])
        if dump:
            with open("game_history.pickle", "wb") as ffile:
                pickle.dump(self, ffile)
        print("data dumped")
        return all_boards, all_shifted_scores, all_discounted_scores, all_move_nrs

    def get_rewards_data(self):
        all_boards = []
        all_scores = []
        for game in self.games:
            scores = [rnd.score for rnd in game.rounds]
            boards = [rnd.board for rnd in game.rounds]
            all_scores.append(scores)
            all_boards.append(boards)
        return all_boards, all_scores

    def get_best_worst_score(self):
        return self.games[-1].final_score, self.games[0].final_score

    def erase(self):
        self.games = []

    @classmethod
    def retrieve(cls):
        gh = None
        with open("game_history.pickle", "rb") as ffile:
            gh = pickle.load(ffile)
        return gh


def shuffle_moves():
    moves = all_moves.copy()
    shuffle(moves)
    return moves
