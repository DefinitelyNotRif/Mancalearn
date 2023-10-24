"""
Handles the base game functions.
"""
from contextlib import contextmanager
import time
import numpy as np


"""
#################################
#     12  11  10  9   8   7     #
#  13                        6  #
#     0   1   2   3   4   5     #
#################################
"""


def validate_input(msg: str, cond_lst: list, final_func=None):
    """
    Asks the user for an input, and then validates it according to the specified conditions. If the input doesn't meet
    any of them, the request is repeated until a valid input is given.
    :param msg: The initial message, asking the user for their input.
    :param cond_lst: A list of lists/tuples of length 2. The first of each pair is a condition (a function that should
    evaluate to True), and the second is the error message to be displayed if the condition evaluates to False. The
    conditions are checked in the order of the list.
    :param final_func: Optionally, this function will be applied to the input
    after all the checks pass. For example, after checking that the user input an integer, you can cast the input string
    to an int.
    :return: The user's input (after final_func is applied).
    """
    print(msg)
    user_input = None
    input_is_valid = False
    conds, c_msgs = zip(*cond_lst)
    # Argument validation
    if not all([callable(c) for c in conds]):
        raise ValueError('Invalid condition list. The first element of each pair must be a function. ')
    if not all([isinstance(m, str) for m in c_msgs]):
        raise ValueError('Invalid condition list. The second element of each pair must be a string. ')

    while not input_is_valid:
        input_is_valid = True
        user_input = input()
        for i in range(len(cond_lst)):  # Check all conditions
            if not conds[i](user_input):  # If the condition doesn't pass
                input_is_valid = False
                print(c_msgs[i], end='\n')
                break

    if user_input is None:  # Shouldn't ever trigger
        raise RuntimeError('Couldn\'t save user input while attempting to validate. ')
    if final_func is not None:
        user_input = final_func(user_input)
    return user_input


def numeric(x, only_int=False) -> bool:
    """
    Determines whether the given input (string or number) can evaluate to a number.
    :param x: The input to be checked.
    :param only_int: If True, returns True only if x can evaluate to an int.
    :return: Whether the input is numeric (True/False).
    """
    try:
        if only_int:
            int(x)
        else:
            float(x)  # Covers both int and float!
        return True
    except ValueError:
        return False


def gen_new_board(board_type='new') -> np.ndarray:
    """
    Creates a new board in the starting layout.
    :param board_type: The layout of the board. 
    'new': The conventional starting board, with 4 seeds in each plot and 0 seeds in the banks. 
    'fast': Same as 'new', but with 1 seed in each plot instead. 
    'instawin': Player 2 has the conventional layout, but player 1's plots are set up so that they may end the game in 
    one turn. That is, the number of seeds in each plot is descending from 6 to 1.  
    :return: A numpy array of length 14. See display(True).
    """
    if board_type == 'new':
        return np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
    elif board_type == 'fast':
        return np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])
    elif board_type == 'instawin':
        return np.array([6, 5, 4, 3, 2, 1, 0, 4, 4, 4, 4, 4, 4, 0])
    else:
        raise ValueError('Invalid board type while attempting to generate. ')


def display(board: np.ndarray, reminder: bool = False):
    """
    Prints a representation of the board (the number of seeds in each plot).
    :param board: The board to be printed.
    :param reminder: If True, lists the index of each plot in the board instead of the number of seeds.
    """
    # TODO: Data validation for the board?
    if reminder:
        board = np.linspace(0, 13, 14, dtype=int)
    b_string = [f'{str(x)} ' if x < 10 else str(x) for x in board]
    print('#################################')
    print(f'#     {"  ".join(b_string[12:6:-1])}    #')
    print(f'#  {b_string[13]}                        {b_string[6]} #')
    print(f'#     {"  ".join(b_string[0:6])}    #')
    print('#################################')


def opposing(pos: int) -> int:  # TODO: Delete?!
    """
    Either finds the opposing plot to the given one.
    :param pos: The plot in question.
    :return: The index of the opposing plot to pos (int).
    """
    if pos < 0 or pos >= 13 or pos == 6:
        raise ValueError(f'Tried to check opposition from an illegal plot ({pos}). ')
    return 12 - pos


def move(board: np.ndarray, player: int, start: int) -> tuple[np.ndarray, int]:
    """
    Plays a move - that is, start sowing seeds from the desired starting plot.
    TODO: Describe rules.
    :param board: The board state before the move.
    :param player: 1 or 2. Player 1 owns plots 0-6, player 2 owns plots 7-13.
    :param start: The index of the plot to start sowing from.
    :return: The board state after the move, and the player that will move next.
    """
    if start < 0 or start >= 13 or start == 6:  # Illegal plot
        raise ValueError(f'Tried to move from an illegal plot ({start}). ')
    if player not in {1, 2}:
        raise ValueError(f'Invalid player ({player}). ')
    # TODO: Data validation for the board?

    new_board = board.copy()  # The board state after the move
    new_player = 3 - player  # The player to play next
    seeds = new_board[start]  # How many seeds to sow
    passed_opp_store = 0  # How many times the player passed the opponent's store in this move
    new_board[start] = 0
    for i in range(start+1, start+1+seeds):
        if (player == 1 and i == 13) or (player == 2 and i == 6):
            passed_opp_store += 1
        new_board[(i + passed_opp_store) % 14] += 1
    stop = (start + seeds + passed_opp_store) % 14

    # Check if the player should go again
    if player == 1 and stop == 6:
        new_player = 1
    if player == 2 and stop == 13:
        new_player = 2

    # Check if the player captured the opponent's seeds
    if player == 1 and stop < 6 and new_board[stop] == 1 and new_board[12 - stop] > 0:  # P1 captures
        new_board[6] += new_board[12 - stop] + 1
        new_board[stop] = 0
        new_board[12 - stop] = 0
    if player == 2 and 6 < stop < 13 and new_board[stop] == 1 and new_board[12 - stop] > 0:  # P2 captures
        new_board[13] += new_board[12 - stop] + 1
        new_board[stop] = 0
        new_board[12 - stop] = 0

    return new_board, new_player


def move_and_disp(board: np.ndarray, player: int, start: int):
    """
    Plays a move and then displays the board.
    :param board: The board state before the move.
    :param player: 1 or 2. Player 1 owns plots 0-6, player 2 owns plots 7-13.
    :param start: The index of the plot to start sowing from.
    :return: The board state after the move.
    """
    newb, newp = move(board, player, start)
    display(newb)
    print(newp)
    return newb


def score(board: np.ndarray) -> tuple[int, int]:
    """
    Returns the scores according to the board state.
    :param board: The board in question.
    :return: A tuple, the first element of which is the score of P1, and the second is the score of P2.
    """
    return board[6], board[13]


def game_over(board: np.ndarray) -> bool:
    """
    Determines whether the game is over. That is, if at least one of the sides is completely empty.
    :param board: The board state in question.
    :return: True if the game is over in the current board state, otherwise False.
    """
    return not(any(board[:6])) or not(any(board[7:13]))


def calc_final_score(board: np.ndarray) -> tuple[int, int]:
    """
    When the game is over, calculates the final score of each player.
    :param board: The board state in question.
    :return: A tuple, the first element of which is the score of P1, and the second is the score of P2.
    """
    return sum(board[:7]), sum(board[7:])


def player_range(player_no: int) -> range:
    """
    Returns the integer range corresponding to the player number entered.
    :param player_no: 1 or 2.
    :return: The plot numbers allotted to the player in question.
    """
    if player_no == 1:
        return range(0, 6)
    elif player_no == 2:
        return range(7, 13)
    else:
        raise ValueError(f'Attempted to get plot range for invalid player ({player_no}). ')


def text_to_board():
    """
    Asks the user to paste the display of the board, and converts it into a numpy array that represents it.
    :return: The entered board in array format.
    """
    print('Paste the board, and then press enter twice to finish. ')
    rows = []
    done = False
    while not done:
        new_row = input()
        if new_row == '':
            done = True
        else:
            rows.append(new_row)
    rows = [[x for x in r.split(' ') if x not in {'', '#'}] for r in rows]
    numbers = [[int(x) for x in r] for r in rows[1:-1]]
    return np.array([*numbers[0], numbers[1][0], *numbers[2], numbers[1][1]])


@contextmanager
def timer():
    start_time = time.time()
    try:
        yield
    finally:
        print(f'Total runtime: {time.time() - start_time} seconds. ')


if __name__ == '__main__':
    # b = gen_new_board()
    # display(b)
    # b = move_and_disp(b, 1, 4)
    # b = move_and_disp(b, 2, 7)
    # b = move_and_disp(b, 1, 0)
    # print(calc_final_score(np.array([1, 2, 3, 4, 5, 6, 20, 0, 0, 0, 0, 0, 0, 10])))
    b = (text_to_board())
    print(b)
