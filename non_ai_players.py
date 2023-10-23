import numpy as np
import gameplay


class RandomPlayer:
    """
    Attributes:
        player_no: 1 or 2, denoting the plots that this player controls.
    Methods:
        move(self, board): Returns the board state after moving. This player selects their move randomly.
    """
    def __init__(self, player_no):
        self.player_no = player_no

    def move(self, board) -> tuple[np.ndarray, int, int, bool]:
        """
        Input: The given board state.
        Outputs:
        new_board: The board state after executing the move.
        decision: The plot that the player decided to move from.
        next_player: The player that will move next (1 or 2).
        is_over: Whether the game is over.
        """
        allowed_range = gameplay.player_range(self.player_no)
        decision = np.random.randint(allowed_range.start, allowed_range.stop)  # 0-5 (if P1) or 7-12 (if P2)
        while board[decision] == 0:  # Don't choose an empty plot!
            decision = np.random.randint(allowed_range.start, allowed_range.stop)
        new_board, next_player = gameplay.move(board, self.player_no, decision)
        is_over = gameplay.game_over(new_board)
        return new_board, decision, next_player, is_over


class HumanPlayer:
    """
    Attributes:
        player_no: 1 or 2, denoting the plots that this player controls.
    Methods:
        move(self, board): Asks the user which plot number to move from, and executes the move that was given.
    """
    def __init__(self, player_no):
        self.player_no = player_no

    def move(self, board) -> tuple[np.ndarray, int, int, bool]:
        """
        Input: The given board state.
        Outputs:
        new_board: The board state after executing the move.
        decision: The plot that the player decided to move from.
        next_player: The player that will move next (1 or 2).
        is_over: Whether the game is over.
        """
        # Input validation
        allowed_range = gameplay.player_range(self.player_no)  # Either 0-5 (if P1) or 7-12 (if P2)
        input_is_valid = False
        decision = -1
        while not input_is_valid:
            input_is_valid = True
            try:
                decision = input(f'Please enter your next move ({allowed_range.start}-{allowed_range.stop-1}): ')
            except KeyboardInterrupt:
                print('hi')
            try:
                decision = int(decision)
            except ValueError as e:  # If the entered value is not an integer
                print(f'Please enter an integer (your input: \"{decision}\"). ')
                input_is_valid = False
            else:  # If it is an integer, check if it's in the range.
                if decision in allowed_range:
                    if board[decision] == 0:
                        print('Please choose a non-empty plot. ')
                        input_is_valid = False
                else:
                    print(f'Please enter an integer in the specified range. ')
                    input_is_valid = False
        assert decision != -1  # Just in case...

        # Execute the move
        new_board, next_player = gameplay.move(board, self.player_no, decision)
        is_over = gameplay.game_over(new_board)
        return new_board, decision, next_player, is_over
