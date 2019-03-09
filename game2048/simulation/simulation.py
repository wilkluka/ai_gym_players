from random import random
from typing import List

from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm
import numpy as np

from bot.solver import BoardSolver
from game_engine.game import Game, GamesHistory, all_moves, shuffle_moves, PastGame, GameRound, GameProgress, Move


def gen_from_beta(beta_par: float):
    """
    for beta_par=1 returns generator for X ~ U(0, 1)
    else returns X ~ Beta(1, ceil(beta_par))
    """
    return np.random.beta(1, np.ceil(beta_par))


def play_with_game_tree(game_state, depth=5):
    if depth:
        move_scores = []
        new_game = game_state.__copy__()
        for move in all_moves:
            if new_game.move(move):
                move_scores.append((move, play_with_game_tree(new_game, depth - 1)[1]))
                new_game = game_state.__copy__()
        if move_scores:
            best_move, best_score = max(move_scores, key=lambda x: x[1])
        else:
            best_move, best_score = shuffle_moves(), 0
    else:
        best_score = game_state.get_board_value()
        best_move = None
    return best_move, best_score


class Simulation(object):
    def __init__(self, expected_rounds=100, history_limit=10):
        self.games_history = GamesHistory("simulation_history", history_limit)
        self.expected_rounds = expected_rounds
        self.model = BoardSolver()
        self.current_max = 1

    def run(self):
        while not self.is_over():
            self.play()
            self.train_solver()
            self.print_summary()
        self.save_model()

    def play(self):
        # self.games_history.erase()
        temp_games_history = GamesHistory("turn_history")
        for _ in tqdm(range(self.expected_rounds)):
            past_game = self.play_round()
            temp_games_history.add_game(past_game)
        temp_games_history.print()
        self.games_history.update(temp_games_history)
        self.games_history.print()

    def play_round(self) -> PastGame:
        random_move_prob = gen_from_beta(self.current_max)
        game = Game()
        past_game = PastGame()
        round_count = 0
        print("random_prob", random_move_prob)
        while not game.is_game_over():
            p = random()  # U(0,1)
            board = game.board.copy()  # for history
            # add if only one choice possible then make it
            if p > random_move_prob:
                possibilities = [possibility for possibility in game.possible_moves.values() if possibility.is_possible]
                choices = self.predict_move(possibilities, game)
            # elif p > random_move_prob:
            #     choices = [play_with_game_tree(game)[0]]
            else:
                choices = shuffle_moves()
            for choice in choices:
                if game.move(choice):
                    past_game.add_round(GameRound(round_count, choice, board))
                    break
            round_count += 1
        past_game.add_round(GameRound(round_count, None, game.board.copy()))  # possible?
        self.current_max = max(self.current_max, past_game.compute_score())
        # past_game.print()
        return past_game

    def print_games(self):
        pass

    def train_solver(self):
        self.model.train(self.games_history)

    def print_summary(self):
        pass

    def save_model(self):
        pass
        # with open(HISTORY_FILENAME, 'wb') as f:
        #     pickle.dump(player.game_history, f)
        # model = player.model.model
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")

    def is_over(self) -> bool:
        return False

    def predict_move(self, possibilities: List[GameProgress], game: Game) -> List[Move]:
        expected_values = self.model.predict(possibilities).reshape(-1)
        curr_value = game.board.value()
        pos_values = np.array([pos.board.value() for pos in possibilities])
        utility_values = pos_values - curr_value + expected_values
        best_possibility_index: int = np.argmax(utility_values)
        return [possibilities[best_possibility_index].move]
