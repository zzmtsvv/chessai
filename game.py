import chess
import chess.uci
from tqdm import tqdm
import numpy as np
import chess.pgn

def expect_elo(Ra, Rb):
    return 1 / (1 + 10 ** ((Rb - Ra) / 400))

def update_elo(Ra, Rb, res):
    Ra_new = Ra + 30 * (res[0] - expect_elo(Ra, Rb))
    Rb_new = Rb + 30 * (res[1] - expect_elo(Rb, Ra))
    return Ra_new, Rb_new

def make_game(engine1, engine2):
    engines = [engine1, engine2]
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    i = 0
    counter = 0

    while not node.board().is_game_over() and counter < 1500:
      engines[i].position(node.board())
      str_move = engines[i].go(movetime=1)[0].__str__()
      node = node.add_main_variation(chess.Move.from_uci(str_move))

      i = 1 - i
      counter += 1

    result = [0, 0]
    if node.board().is_checkmate():
      winner = 1 - i
      result[1 - i] = 1

    if node.board().is_stalemate() or node.board().is_fivefold_repetition():
      result = [0.5, 0.5]

    if node.board().is_seventyfive_moves() or counter == 1500:
      result = [0.5, 0.5]

    if node.board().is_insufficient_material():
      result = [0.5, 0.5]

    return result, game
