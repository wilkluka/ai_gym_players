from enum import Enum
from bisect import insort
import numpy as np
from functools import total_ordering
from random import shuffle

BOARD_SIZE = 4


class Move(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


all_moves = list(Move.__members__.values())


def get_empty_board() -> np.array:
    return np.zeros([BOARD_SIZE, BOARD_SIZE], dtype='uint16')


class GameProgress:

    def __init__(self, move, is_possible, board=None):
        self.move = move
        self.is_possible = is_possible
        self.board = board if board is not None else get_empty_board()

    def __hash__(self) -> int:
        return int(get_board_value(self.board)) + hash(self.move)


def get_board_value(board) -> float:
    return sum(v * (np.log2(max(v, 1)) - 1) for v in board.flatten())


class GameRound:

    def __init__(self, game_round_num, move, board=None):
        self.move = move
        self.game_round_num = game_round_num
        self.board = board if board is not None else get_empty_board()

    def print(self):
        print(self.board)
        print(self.move)



@total_ordering
class PastGame:

    def __init__(self):
        self.score = 0
        self.final_score = 0
        self.rounds = []

    def add_round(self, game_round: GameRound):
        self.rounds.append(game_round)

    def __eq__(self, other) -> bool:
        return self.score == other.score

    def __lt__(self, other) -> bool:
        return self.score < other.score

    def compute_score(self):
        self.score = self.final_score = get_board_value(self.rounds[-1].board)

    def print(self):
        for roundd in self.rounds:
            roundd.print()


def are_boards_equal(left, right):
    return np.all(left == right)


class Game:

    def __init__(self):
        self.board = get_empty_board()
        self._add_new_board_tile()
        self._add_new_board_tile()
        self.possible_moves = {move: GameProgress(move, True) for move in all_moves}
        self._refresh_possible_moves()

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
        self.possible_moves[move].is_possible = not are_boards_equal(self.board, self.possible_moves[move].board)

    def _refresh_possible_moves(self):
        for move in all_moves:
            self._make_move(move)

    def move(self, move):
        if not self.possible_moves[move].is_possible:
            return False

        self._set_board(self.possible_moves[move].board, add_new_tile=True, refresh_moves=True)

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

    def __init__(self, limit=1000):
        self.games = []
        self.limit = limit

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
        for game_n in range(1, last_n+1):
            game = self.games[-game_n]
            print(game.score, '\t', game.rounds[-1].game_round_num, '\t', game.rounds[-1].move)
            print(game.rounds[-1].board)
            print('===========')
        print('#########################')

    def get_training_data(self):
        move_boards = sum([[(game_round.move, game_round.board) for game_round in game.rounds] for game in self.games], [])
        moves, boards = map(list, zip(*move_boards))
        return moves, boards

    def erase(self):
        self.games = []

def shuffle_moves():
    moves = all_moves.copy()
    shuffle(moves)
    return moves
