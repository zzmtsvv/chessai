import chess
import chess.pgn
import numpy as np
import pandas as pd

def str2coord(move):
  translation_table = str.maketrans('abcdefgh', '01234567')
  return 8 - int(move[1]), int(move[0].translate(translation_table))

def coord2str(y, x):
  translation_table = str.maketrans('01234567', 'abcdefgh')
  return str(x).translate(translation_table) + str(8 - y)

def ohe_board(board, white=True):
  if white:
    translation_table = str.maketrans('BKNPQRbknpqr.', '0123456666666')
  else:
    translation_table = str.maketrans('bknpqrBKNPQR.', '0123456666666')
  
  ohe_map = np.array([[1., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0.]])
  
  text_board = board.__str__()
  text = text_board.translate(translation_table).replace('\n', ' ').split(' ')
  int_flatten_arr = np.array(text).astype('int')

  return ohe_map[int_flatten_arr].reshape((8, 8, 6))

def flip_move(move):
  y0, x0 = str2coord(move[:2])
  y1, x1 = str2coord(move[2:4])
  y0 = 7 - y0
  y1 = 7 - y1
  return coord2str(y0, x0) + coord2str(y1, x1)

def get_position(game, move_number):
  counter = 0
  for i in game.mainline():
    if counter == move_number:
      break
    counter += 1
  return i

def get_position_tensor(board, white=True):
  pass
