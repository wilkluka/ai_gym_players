import pytest
import numpy as np

from game2048.game_engine.game import Game, Move, PastGame, GameProgress, Board


@pytest.mark.parametrize(
    'board,possible_moves,impossible_moves',
    [
        (
                [[2, 8, 4, 2],
                 [8, 64, 8, 16],
                 [2, 32, 2, 4],
                 [4, 8, 4, 4]],
                [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT],
                []
        ),
        (
                [[2, 8, 4, 2],
                 [8, 64, 8, 16],
                 [2, 32, 2, 2],
                 [4, 8, 4, 4]],
                [Move.RIGHT, Move.LEFT],
                [Move.UP, Move.DOWN]
        ),
        (
                [[0, 16, 0, 2],
                 [0, 8, 4, 2],
                 [0, 2, 32, 4],
                 [2, 8, 4, 16]],
                [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT],
                []
        ),
        (
                [[2, 8, 4, 2],
                 [2, 64, 8, 16],
                 [2, 32, 4, 2],
                 [4, 8, 2, 4]],
                [Move.UP, Move.DOWN],
                [Move.RIGHT, Move.LEFT]
        ),
        (
                [[0, 8, 4, 2],
                 [2, 64, 8, 16],
                 [2, 32, 4, 2],
                 [4, 8, 2, 4]],
                [Move.UP, Move.LEFT, Move.DOWN],
                [Move.RIGHT]
        ),
        (
                [[0, 8, 4, 2],
                 [2, 64, 8, 16],
                 [16, 32, 4, 2],
                 [4, 8, 2, 4]],
                [Move.UP, Move.LEFT],
                [Move.RIGHT, Move.DOWN]
        )

    ]
)
def test_possible_and_impossible_moves(board, possible_moves, impossible_moves):
    game = Game()
    game.set_from_other(Board(board))
    for move, game_progress in game.possible_moves.items():
        if move in possible_moves:
            assert game_progress.is_possible
        else:
            assert not game_progress.is_possible


@pytest.mark.parametrize(
    'initial_board,game_progresses',
    [
        [
            [[0, 8, 4, 2],
             [2, 64, 8, 16],
             [4, 32, 4, 2],
             [4, 8, 2, 4]],
            [
                GameProgress(
                    Move.UP,
                    True,
                    Board([[2, 8, 4, 2],
                           [8, 64, 8, 16],
                           [0, 32, 4, 2],
                           [0, 8, 2, 4]])
                ),
                GameProgress(
                    Move.DOWN,
                    True,
                    Board([[0, 8, 4, 2],
                           [0, 64, 8, 16],
                           [2, 32, 4, 2],
                           [8, 8, 2, 4]])
                ),
                GameProgress(
                    Move.RIGHT,
                    True,
                    Board([[0, 8, 4, 2],
                           [0, 64, 8, 16],
                           [2, 32, 4, 2],
                           [0, 16, 2, 4]])
                ),
                GameProgress(
                    Move.LEFT,
                    True,
                    Board([[0, 8, 4, 2],
                           [0, 64, 8, 16],
                           [2, 32, 4, 2],
                           [16, 2, 4, 0]])
                ),
                GameProgress(
                    Move.UP,
                    True,
                    Board([[2, 8, 4, 2],
                           [16, 64, 8, 16],
                           [0, 32, 8, 2],
                           [0, 2, 0, 0]])
                )

            ]
        ]
    ]
)
def test_game_progression(initial_board, game_progresses):
    game = Game()
    game.set_from_other(Board(initial_board))
    for game_progress in game_progresses:
        assert game_progress in game.possible_moves.values()
        game.set_from_other(game_progress.board)
