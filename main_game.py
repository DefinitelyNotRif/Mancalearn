import os
import time
from datetime import datetime
import numpy as np
import gameplay
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


def save_weights(player, info=None):
    """
    Saves the weights of the given player.
    The function will create a designated folder, the name of which is the time of its creation and number of rounds.
    There, each weight will be saved in a separate .csv file, with the names w1, b1, w2, b2, etc.
    In addition, a 'game_info.csv' file will be created, detailing the player types and number of rounds.
    :param player: The player to extract the weights from. Cannot be from non_ai_players!
    :param info: The game info (for game_info.csv), as a list of length 3.
    """
    # Input validation
    if not isinstance(player, reinforcement_player.ReinforcementPlayer):
        raise ValueError(f'Cannot save weights of non-AI player ({type(player)})!')
    # Create the folder
    folder_str = str(datetime.now()).replace(':', '_')
    if info is not None:
        folder_str = f'({info[2]}) {folder_str}'
    os.mkdir(f'weights_results/{folder_str}')
    # Create the weight files
    player_weights = player.q_network.get_weights()
    for num, weight in enumerate(player_weights):
        if weight.ndim == 1:
            w_type = 'b'
        elif weight.ndim == 2:
            w_type = 'w'
        else:
            raise ValueError(f'Invalid weight shape ({weight.ndim})!')
        np.savetxt(f'weights_results/{folder_str}/{w_type}{num // 2 + 1}.csv', weight, delimiter=',')
    if info is not None:
        # Create the info file
        lines = [f'Player 1: {info[0]}\n'
                 f'Player 2: {info[1]}\n'
                 f'Number of rounds: {info[2]}']
        with open(f'weights_results/{folder_str}/game_info.txt', 'w') as f:
            f.writelines(lines)


def list_games(limit: int = None, prompt: bool = False):
    """
    Prints a numbered list of all the folders containing weight data. If specified, the user can enter a number, and
    the name of that folder will be returned.
    :param limit: The maximum number of folders to print.
    :param prompt: If True, prompts the user to choose a number from the list, and gets that folder's name.
    :return: If prompt is True, returns the name of the selected folder. Otherwise, returns None.
    """
    folder_list = os.listdir('weights_results')
    if limit is None:
        limit = len(folder_list)
    else:
        limit = min(len(folder_list), limit)
    for i in range(limit):
        print(f'[{i + 1}] {folder_list[i]}')
    if prompt:
        user_choice = input(f'Enter the number of the folder you want to select (1-{limit}): ')
        valid_choice = False
        while not valid_choice:
            try:
                user_choice = int(user_choice)
                if user_choice in range(1, limit+1):
                    valid_choice = True
                else:
                    user_choice = input(f'Please enter a number in the range 1-{limit}. ')
            except ValueError:
                user_choice = input(f'Please enter an integer in the specified range. ')
        return folder_list[user_choice - 1]


def load_weights(folder_name, player, reset_epsilon=False):
    """
    Gets the weights saved in the specified folder, and loads them into the given player.
    :param folder_name: The folder (in weights_results) of the desired weights.
    :param player: The ReinforcementPlayer to load the weights into.
    :param reset_epsilon: If True, the AI player will start with epsilon = 1. Else, set it to
    its minimum value.
    """
    folder_path = f'weights_results/{folder_name}'
    file_list = os.listdir(folder_path)
    file_list.remove('game_info.txt')
    num_layers = int(len(file_list) / 2)
    w_lst = [np.array([])] * num_layers
    b_lst = [np.array([])] * num_layers
    for filename in file_list:
        weights = np.loadtxt(f'{folder_path}/{filename}', delimiter=',')
        prefix = filename[0]
        num = int(filename[1])
        if prefix == 'w':
            w_lst[num - 1] = weights
        elif prefix == 'b':
            b_lst[num - 1] = weights
        else:
            raise ValueError(f'Unexpected file encountered ({filename}). ')
    assert all([len(a) > 0 for a in w_lst])
    assert all([len(a) > 0 for a in b_lst])
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
        save_weights(p1, info=game_info)


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
    # human_vs_ai(1, True, False)
    ai_vs_random(100, from_file=True)

    # print(list_games(10, True))

    # p1 = non_ai_players.HumanPlayer(1)
    # p2 = reinforcement_player.ReinforcementPlayer(2)
    # folder_name = list_games(prompt=True)
    # load_weights(folder_name, p2)
    # run_game(p1, p2, print_pbp=True, print_scores=True)
