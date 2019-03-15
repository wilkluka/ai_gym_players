from random import random
from typing import List

from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import subprocess

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
        self.temp_games_history = None
        self.tensorboardx_writer = SummaryWriter("logs/game_results")

    def run(self):
        ticker = tqdm(total=100, desc="total turns")
        while not self.is_over():
            ticker.update(1)
            self.play()
            self.tensorboardx_writer.add_scalar(
                "best_score",
                self.games_history.games[-1].final_score,
                global_step=ticker.n
            )
            self.tensorboardx_writer.add_scalar(
                "best_score_this_episode",
                self.temp_games_history.games[-1].final_score,
                global_step=ticker.n
            )
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
        while not game.is_game_over():
            p = random()  # U(0,1)
            # current_board = game.board.copy()
            if p > random_move_prob:
                possibilities = [possibility for possibility in game.possible_moves.values() if possibility.is_possible]
                choices = self.predict_move(possibilities, game)
            # elif p > random_move_prob:
            #     choices = [play_with_game_tree(game)[0]]
            else:
                choices = shuffle_moves()
            for choice in choices:
                if game.possible_moves[choice].is_possible:
                    past_game.add_round(GameRound(round_count, choice, game.possible_moves[choice].board.copy()))
                    # past_game.add_round(GameRound(round_count, choice, current_board))
                    game.move(choice)
                    break
            round_count += 1
        past_game.add_round(GameRound(round_count, None, game.board.copy()))  # possible?
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
        return False

    def predict_move(self, possibilities: List[GameProgress], game: Game) -> List[Move]:
        IMMEDIATE_REWARD_MULT = 2
        expected_values = self.model.predict(possibilities, game.move_count).reshape(-1)
        curr_value = game.board.value()
        pos_values = np.array([pos.board.value() for pos in possibilities])
        utility_values = (pos_values - curr_value) * IMMEDIATE_REWARD_MULT + expected_values
        best_possibility_index = int(np.argmax(utility_values))
        return [possibilities[best_possibility_index].move]
