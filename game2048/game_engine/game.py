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
    # def value(self) -> int:
    #     return int(self.max() + self.sum())

    @staticmethod
    def get_empty():
        return Board(np.zeros([BOARD_SIZE, BOARD_SIZE], dtype='uint16'))

    def __repr__(self):
        return "Board({})".format(self.tolist())


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
        self.final_score = self.rounds[-1].board.value()
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
            return
        value_to_insert = np.random.choice(a=[2, 4], size=1, p=[0.9, 0.1])
        insert_point = np.random.choice(np.arange(0, BOARD_SIZE), 2)
        while self.board[tuple(insert_point)] != 0:
            insert_point = np.random.choice(np.arange(0, BOARD_SIZE), 2)
        self.board[tuple(insert_point)] = value_to_insert

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
        new_ob = GamesHistory("merged")
        new_ob.games = first.games + second.games
        return new_ob

    def add_game(self, past_game: PastGame):
        insort(self.games, past_game)
        if len(self.games) > self.limit:
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

    def get_training_data_move_board(self):
        move_boards = sum([[(game_round.move, game_round.board) for game_round in game.rounds] for game in self.games],
                          [])
        moves, boards = map(list, zip(*move_boards))
        return moves, boards

    def get_training_data_board_reward(self):
        all_boards = []
        all_rewards = []
        all_move_nrs = []
        # discount = 0.9
        reward_part = 0.3
        for game in self.games:
            move_nrs, scores = map(list, zip(*[(rnd.game_round_num, rnd.score) for rnd in game.rounds]))
            # rewards = [2 * nxt - curr for curr, nxt in zip(board_values, board_values[1:])]
            # curr = rewards[-1]
            # new_rewards = []
            # for r in rewards[::-1]:
            #     curr = curr * discount + r * (1-discount)
            #     new_rewards.append(curr)
            # new_rewards.reverse()

            # this is small hack to ensure that we have sufficient data for higher level boards
            all_rewards.extend(scores[2:] + [scores[-1]] * 2)
            # trim is necessary coz we drop last round for computing deltas at line with zip
            all_boards.extend([game_round.board for game_round in game.rounds])
            all_move_nrs.extend(move_nrs)
            # print(len(all_boards), len(all_rewards))
            assert len(all_boards) == len(all_rewards)
        with open("game_history.pickle", "wb") as ffile:
            pickle.dump((all_boards, all_rewards), ffile)
        print("data dumped")
        return all_boards, all_rewards, all_move_nrs

    def get_best_worst_score(self):
        return self.games[-1].final_score, self.games[0].final_score

    def erase(self):
        self.games = []


def shuffle_moves():
    moves = all_moves.copy()
    shuffle(moves)
    return moves
