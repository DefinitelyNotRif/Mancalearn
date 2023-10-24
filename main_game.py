import os
import time
from datetime import datetime
import numpy as np
import gameplay
from gameplay import numeric, validate_input
import non_ai_players
import reinforcement_player
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def run_game(p1, p2, num_games: int = 1, board_type: str = 'new', print_pbp: bool = False, print_scores: bool = False) \
        -> tuple[list[tuple[int, int]], list[str, str, int]]:
    """
    Executes a full game.
    :param p1, p2: The Player instances that will participate in the game.
    :param num_games: The number of games to run.
    :param board_type: 'new', 'fast', etc.
    :param print_pbp: Stands for "play-by-play". If True, displays the board and prints each player's decision after
    each move.
    :param print_scores: If True, prints the score at the end of each game.
    :return: A list of all the outcomes as tuples.
    """
    players = [p1, p2]
    scores = []
    for _ in range(num_games):
        board = gameplay.gen_new_board(board_type)
        done = False  # Whether the game is over
        to_move = 1  # The player that should move next
        move_no = 0  # The number of moves that have been executed
        if print_pbp:
            gameplay.display(board, True)
            gameplay.display(board, False)
        while not done:
            new_board, decision, new_to_move, done = players[to_move - 1].move(board)
            if print_pbp:
                gameplay.display(new_board)
                print(f'Player {to_move} moved from plot #{decision}. It is now player {new_to_move}\'s turn. ')
                if isinstance(players[to_move - 1], non_ai_players.HumanPlayer) and new_to_move != to_move:
                    time.sleep(1)
            move_no += 1
            board = new_board
            to_move = new_to_move
        for p in players:
            if isinstance(p, reinforcement_player.ReinforcementPlayer):
                p.update_epsilon()
        game_scores = gameplay.calc_final_score(board)
        win_msg = f'Player {np.argmax(game_scores) + 1} wins!' if game_scores[0] != game_scores[1] else 'It\'s a tie! '
        if print_scores:
            print(f'The final score is {game_scores[0]}-{game_scores[1]}. {win_msg}')
        if print_pbp:
            print(f'The game took {move_no} moves. ')
        scores.append(game_scores)
    game_info = [p1.__class__.__name__, p2.__class__.__name__, num_games]
    return scores, game_info


def save_weights(player, info=None, comment=None):
    """
    Saves the weights of the given player.
    The function will create an .npz file, the name of which is the time of its creation and number of rounds.
    There, each weight will be saved as a separate array, with the names w1, b1, w2, b2, etc.
    In addition, the player types, number of rounds and additional comments may be saved in a 'game_info' array.
    :param player: The player to extract the weights from. Cannot be from non_ai_players!
    :param info: A list of length 3, containing the type of each player (as a string) and the number of rounds.
    :param comment: A description of the game.
    """
    # Input validation
    if not isinstance(player, reinforcement_player.ReinforcementPlayer):
        raise ValueError(f'Cannot save weights of non-AI player ({type(player)})!')

    names_to_save = []
    arrs_to_save = []
    # Add the game info first
    info_arr = []
    if info is not None:
        info_arr.extend(info)
    if comment is not None:
        info_arr.append(comment)
    if len(info_arr) > 0:
        names_to_save.append('game_info')
        arrs_to_save.append(info_arr)

    # Add the weights
    player_weights = player.q_network.get_weights()
    for num, weight in enumerate(player_weights):
        if weight.ndim == 1:
            w_type = 'b'
        elif weight.ndim == 2:
            w_type = 'w'
        else:
            raise ValueError(f'Invalid weight shape ({weight.ndim})!')
        w_num = num // 2 + 1
        names_to_save.append(f'{w_type}{w_num}')
        arrs_to_save.append(weight)

    kwargs_dict = dict(zip(names_to_save, arrs_to_save))
    filename = str(datetime.now()).replace(':', '_')
    filename = f'weights_results/{filename} ({info[2]})'
    np.savez(filename, **kwargs_dict)


def list_games(limit: int = None, with_comments: bool = False, prompt: bool = False):
    """
    Prints a numbered list of all the files containing weight data. If specified, the user can enter a number, and
    the name of that file will be returned.
    :param limit: The maximum number of folders to print.
    :param with_comments: If True, appends the comment (if it exists) to each filename.
    :param prompt: If True, prompts the user to choose a number from the list, and gets that folder's name.
    :return: If prompt is True, returns the name of the selected folder. Otherwise, returns None.
    """
    file_list = os.listdir('weights_results')  # A list of all the file names
    # Set the display limie
    if limit is None:
        limit = len(file_list)
    else:
        limit = min(len(file_list), limit)
    # Print the file names, and comments if specified
    for i in range(limit):
        to_print = f'[{i + 1}] {file_list[i]}'  # The numbered file names
        if with_comments:
            game_file = np.load(f'weights_results/{file_list[i]}')  # NpzFile object
            if 'game_info' in game_file:  # If the file contains a comment
                comment = game_file['game_info'][-1]  # Isolate the comment itself
                to_print = f'{to_print} ({comment})'  # And append it to the string
        print(to_print)
    if prompt:
        user_choice = validate_input(f'Enter the number of the folder you want to select (1-{limit}): ',
                                     [[lambda x: numeric(x, True),
                                       'Please enter an integer in the specified range. '],
                                      [lambda x: int(x) in range(1, limit+1),
                                      f'Please enter a number in the range 1-{limit}. ']],
                                     int)  # The 'int' at the end is the casting function!
        return file_list[user_choice - 1]


def load_weights(filename, player, reset_epsilon=False):
    """
    Gets the weights saved in the specified file, and loads them into the given player.
    :param filename: The file (in weights_results) containing the desired weights.
    :param player: The ReinforcementPlayer to load the weights into.
    :param reset_epsilon: If True, the AI player will start with epsilon = 1. Else, set it to
    its minimum value.
    """
    game_npz = np.load(f'weights_results/{filename}')
    num_layers = int((len(game_npz) - 1) / 2)
    w_lst = [game_npz[f'w{i+1}'] for i in range(num_layers)]
    b_lst = [game_npz[f'b{i + 1}'] for i in range(num_layers)]
    player.set_weights(w_lst, b_lst)
    if not reset_epsilon:
        player.epsilon = player.e_min


def ai_vs_ai(num_games):
    p1 = reinforcement_player.ReinforcementPlayer(1)
    p2 = reinforcement_player.ReinforcementPlayer(2)
    with gameplay.timer():
        print('Starting...')
        repeated_scores, game_info = run_game(p1, p2, num_games)
        winners = [np.argmax(tup) + 1 for tup in repeated_scores]
    print(repeated_scores)
    for i in {1, 2}:
        print(f'Player {i} won {winners.count(i)} times. ')

    winner_player = p1 if winners.count(1) > winners.count(2) else p2
    q_weights = winner_player.q_network.get_weights()
    save_weights(winner_player, info=game_info)


def ai_vs_random(num_games, from_file=False, save=False):  # AI is always P1 for now
    p1 = reinforcement_player.ReinforcementPlayer(1)
    p2 = non_ai_players.RandomPlayer(2)
    if from_file:
        folder_name = list_games(prompt=True)
        load_weights(folder_name, p1)
    with gameplay.timer():
        print('Starting...')
        repeated_scores, game_info = run_game(p1, p2, num_games)
        winners = [np.argmax(tup) + 1 for tup in repeated_scores]
    print(repeated_scores)
    for i in {1, 2}:
        print(f'Player {i} won {winners.count(i)} times. ')

    if save:
        comment = input('Enter a description: ')
        save_weights(p1, info=game_info, comment=comment)


def human_vs_ai(human_num=1, from_file=False, save=False):
    if from_file:
        folder_name = list_games(prompt=True)
    if human_num == 1:
        p1 = non_ai_players.HumanPlayer(1)
        p2 = reinforcement_player.ReinforcementPlayer(2)
        if from_file:
            load_weights(folder_name, p2)
    else:
        p1 = reinforcement_player.ReinforcementPlayer(1)
        p2 = non_ai_players.HumanPlayer(2)
        if from_file:
            load_weights(folder_name, p1)
    print('Starting...')
    repeated_scores, game_info = run_game(p1, p2, 1, print_scores=True, print_pbp=True)

    if save:
        ai_player = p1 if human_num == 2 else p2
        save_weights(ai_player, info=game_info)


if __name__ == '__main__':
    pass
    # human_vs_ai(1, True, False)
    # ai_vs_random(10, save=True)

    list_games(2)
    # print(list_games(10, with_comments=True, prompt=True))

    # p1 = non_ai_players.HumanPlayer(1)
    # p2 = reinforcement_player.ReinforcementPlayer(2)
    # player_filename = list_games(prompt=True)
    # load_weights(player_filename, p2)
    # run_game(p1, p2, print_pbp=True, print_scores=True)
