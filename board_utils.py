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
  white_ohe = ohe_board(board, white=True)
  black_ohe = ohe_board(board, white=False)

  if white:
    feature_tensor = np.concatenate([white_ohe, black_ohe], axis=-1)
  else:
    feature_tensor = np.concatenate([black_ohe, white_ohe], axis=-1)
    feature_tensor = np.flip(feature_tensor, axis=0)
  return feature_tensor

def move2tensor(move):
  move_from_str, move_to_str = move[:2], move[2:4]
  move_from = coord2str(move_from_str)
  move_to = coord2str(move_to_str)

  move_tensor = np.zeros((8, 8, 81))
  res = np.zeros(81)
  knife_map = {(2, 1): 0, (2, -1): 1, (1, 2): 2, (1, -2): 3, (-1, 2): 4,
               (-1, -2): 5, (-2, 1): 6, (-2, -1): 7}
  diff_x = move_to[1] - move_from[1]
  diff_y = move_to[0] - move_from[0]

  if not diff_x:
    move_type = 0
    result[move_type * 8 + move_to[0]] = 1
  elif not diff_y:
    move_type = 1
    result[move_type * 8 + move_to[1]] = 1
  elif diff_x == diff_y:
    move_type = 2
    result[move_type * 8 + move_to[1]] = 1
  elif diff_x == -diff_y:
    move_type = 3
    result[move_type * 8 + move_to[1]] = 1
  elif np.abs(diff_x * diff_y) == 2:
    move_type = 4
    result[move_type * 8 + knife_map[diff_x, diff_y]] = 1
  
  move_tensor[move_from[0], move_from[1], :] = result
  return move_tensor

def tensor2move(move_tensor, white=True):
  y, x, num = np.unravel_index(move_tensor.argmax(), move_tensor.shape)
  knife_map = {0: (2, 1), 1: (2, -1), 2: (1, 2), 3: (1, -2),
               4: (-1, 2), 5: (-1, -2), 6: (-2, 1), 7: (-2, -1)}
  move_from_str = coord2str(y, x)
  move_type = num // 8
  if move_type == 0:
    move_to_str = coord2str(num % 8, x)
  elif move_type == 1:
    move_to_str = coord2str(y, num % 8)
  elif move_type == 2:
    move_to_str = coord2str(y + (num % 8 - x), num % 8)
  elif move_type == 3:
    move_to_str = coord2str(y - (num % 8 - x), num % 8)
  elif move_type == 4:
    diff_x, diff_y = knife_map[num % 8]
    move_to_str = coord2str(y + diff_y, x + diff_x)
  
  move_str = move_from_str + move_to_str
  if not white:
    move_str = flip_move(move_str)
  return move_str
