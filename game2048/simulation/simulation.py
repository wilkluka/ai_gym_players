from logging import warning
from random import random
from typing import List

from tqdm import tqdm
import numpy as np
import subprocess

from bot.solver import BoardSolver
from game_engine.game import Game, GamesHistory, all_moves, shuffle_moves, PastGame, GameRound, GameProgress, Move
from utils.magic_collections import MagicList
from utils.tb import TBLogger

REWARD_MULT = 1


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
        self.temp_games_history = None
        self.tb_best = TBLogger("logs/game_results_best")
        self.tb_worst = TBLogger("logs/game_results_worst")
        self.tb_hists = TBLogger("logs/game_results_hists")
        self.episodes_count = 0

    def run(self):
        while not self.is_over():
            self.play()
            self.write_play_logs()
            self.train_solver()
            self.print_summary()
        self.save_model()

    def play(self):
        # self.games_history.erase()
        self.temp_games_history = GamesHistory("turn_history")
        for _ in tqdm(range(self.expected_rounds)):
            past_game = self.play_round()
            self.temp_games_history.add_game(past_game)
        self.temp_games_history.print()
        self.games_history.update(self.temp_games_history)
        self.games_history.print()

    def play_round(self) -> PastGame:
        random_move_prob = gen_from_beta(np.power(self.current_max, 5/4))
        game = Game()
        past_game = PastGame()
        round_count = 0
        print("random_prob", random_move_prob)
        warning("p = 1 which means we will do all moves with NN")
        while not game.is_game_over():
            p = random()  # U(0,1)
            p = 1
            current_board = game.board.copy()
            if p > random_move_prob:
                possibilities = [possibility for possibility in game.possible_moves.values() if possibility.is_possible]
                choices = self.predict_move(possibilities, game)
            # elif p > random_move_prob:
            #     choices = [play_with_game_tree(game)[0]]
            else:
                choices = shuffle_moves()
            for choice in choices:
                if game.possible_moves[choice].is_possible:
                    # past_game.add_round(GameRound(round_count, choice, current_board))
                    game.move(choice)
                    past_game.add_round(GameRound(round_count, choice, game.current_score, current_board))
                    break
            round_count += 1
        last_score = past_game.compute_score()
        if last_score > self.current_max:
            self.current_max = last_score
            subprocess.call("paplay /usr/share/sounds/freedesktop/stereo/complete.oga".split())
        # past_game.print()
        return past_game

    def print_games(self):
        pass

    def train_solver(self):
        self.model.train(GamesHistory.merge(self.games_history, self.temp_games_history))

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
        self.episodes_count += 1
        return False

    def predict_move(self, possibilities: List[GameProgress], game: Game) -> List[Move]:
        expected_values = self.model.predict(possibilities).reshape(-1)
        curr_value = game.board.value()
        pos_values = np.array([pos.board.value() for pos in possibilities])
        utility_values = (pos_values - curr_value) * REWARD_MULT + expected_values
        best_possibility_index = int(np.argmax(utility_values))
        return [possibilities[best_possibility_index].move]

    def write_play_logs(self):
        simul_best, simul_worst = self.games_history.get_best_worst_score()
        episode_best, episode_worst = self.temp_games_history.get_best_worst_score()
        self.tb_best.log_scalar("score/simulation", simul_best, self.episodes_count)
        self.tb_best.log_scalar("score/episode", episode_best, self.episodes_count)

        self.tb_worst.log_scalar("score/simulation", simul_worst, self.episodes_count)
        self.tb_worst.log_scalar("score/episode", episode_worst, self.episodes_count)

        _, episode_rewards = self.temp_games_history.get_training_data()
        _, simul_rewards = self.games_history.get_training_data()
        self.tb_hists.log_histogram(
            "score/simulation",
            MagicList(simul_rewards).flatten(),
            self.episodes_count
        )
        self.tb_hists.log_histogram(
            "score/episode",
            MagicList(episode_rewards).flatten(),
            self.episodes_count
        )
