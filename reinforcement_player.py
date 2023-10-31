import random
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input
from keras.losses import MSE
from keras.optimizers import Adam
import gameplay
from collections import deque


class ReinforcementPlayer:
    """
    TODO detailed description.
    Notes:
        Reward: score gained during this turn. If you land in your bank (i.e. next_player == self.player_no and
        the reward should have been 1), get 2 instead. Illegal moves get -100.
        This should incentivize capturing over all else, then landing in your bank, then getting one point.
        What it DOESN'T do is disincentivize giving the opponent good turns (e.g. letting them capture you).
    """
    def __init__(self, player_no, **kwargs):
        self.player_no = player_no
        # self.player_range = gameplay.player_range(self.player_no)
        self.time = 0
        # kwargs
        self.buffer_size = kwargs.get('buffer_size', int(1e5))
        self.steps_per_update = kwargs.get('steps_per_update', 4)
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        self.gamma = kwargs.get('gamma', 0.99)
        self.alpha = kwargs.get('alpha', 1e-3)
        self.epsilon = kwargs.get('epsilon', 1)
        self.e_decay = kwargs.get('e_decay', 0.995)
        self.e_min = kwargs.get('e_min', 0.01)
        self.tau = kwargs.get('tau', 1e-3)
        self.layer_size = kwargs.get('layer_size', 100)
        self.invalid_reward = kwargs.get('invalid_reward', -100)
        # Initialize memory buffer
        self.memory_buffer = deque(maxlen=self.buffer_size)
        # Initialize Q-network
        state_size = 14  # for each plot/bank in the board
        num_actions = 6  # One for each plot you can move from
        self.optimizer = Adam(learning_rate=self.alpha)
        self.q_network = Sequential([
            Input(state_size),
            Dense(self.layer_size, activation='relu'),
            Dense(self.layer_size, activation='relu'),
            Dense(num_actions, activation='linear')
        ])
        # Initialize Q^-network
        self.target_q_network = Sequential([
            Input(state_size),
            Dense(self.layer_size, activation='relu'),
            Dense(self.layer_size, activation='relu'),
            Dense(num_actions, activation='linear')
        ])
        self.target_q_network.set_weights(self.q_network.get_weights())

    def move(self, board):
        if self.player_no == 2:  # Set the board so that the network thinks it's P1
            board = self.flip_board(board)
        greedy_move, greedy_reward = self._greedy_action(board)
        use_random = random.random()  # If less than epsilon, choose a RANDOM move
        if use_random < self.epsilon:
            decision = random.randint(0, 5)
            while board[decision] == 0:  # Don't try to move from an empty plot
                decision = random.randint(0, 5)
            reward = self._reward(board)[1][decision]
        else:
            decision = greedy_move
            reward = greedy_reward
        *next_state, move_again = self.move_to_state(gameplay.move(board, 1, decision))  # move_again is the LAST element
        # reward = self._reward(board)[decision % 7]
        done = gameplay.game_over(next_state)

        self.memory_buffer.append((board, decision, reward, next_state, done))
        if (self.time + 1) % self.steps_per_update == 0 and len(self.memory_buffer) > self.minibatch_size:
            experiences = random.sample(self.memory_buffer, k=self.minibatch_size)
            experiences = [np.array(x) for x in zip(*experiences)]
            self._agent_learn(experiences)
        self.time += 1
        if self.player_no == 2:  # Flip the board back
            next_state = self.flip_board(next_state)
            decision += 7
        next_player = self.player_no if move_again == 1 else 3 - self.player_no
        return next_state, decision, next_player, done

    @staticmethod
    def move_to_state(move_tup):
        """
        Converts move_tup, a tuple of the form (board, move_again) to a state array of the
        form [b0, b1, ..., b13, move_again].
        move_again is 1 if the player gets another move, and 0 otherwise.
        """
        move_again = 2 - move_tup[1]
        return np.append(move_tup[0], move_again)

    def _compute_loss(self, experiences):
        states, actions, rewards, next_states, done_vals = experiences
        max_qsa = tf.reduce_max(self.target_q_network(next_states), axis=-1)
        y_targets = rewards + (1 - done_vals) * self.gamma * max_qsa
        q_values = self.q_network(states)
        # action_idxs = actions % 7  # Supposedly not necessary anymore
        interim_stack = tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1)
        q_values = tf.gather_nd(q_values, interim_stack)
        return MSE(y_targets, q_values)

    def _agent_learn(self, experiences):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        # Update the target network
        for target_weights, q_weights in zip(self.target_q_network.weights, self.q_network.weights):
            target_weights.assign(self.tau * q_weights + (1 - self.tau) * target_weights)

    def _reward(self, board):
        """
        Returns the immediate reward for each of the six possible moves (along with their respective states).
        TODO: Inefficient! Revisit this with a clear mind.
        """
        # my_bank = 6
        row_before = board[:6]
        hit_bank = np.array([board[i] == 6 - i for i in range(6)])  # Whether
        # a move from this slot reaches the bank exactly
        possible_states = np.apply_along_axis(
            lambda i: self.move_to_state(gameplay.move(board, 1, i[0])), 1,
            np.arange(6).reshape(-1, 1))  # Each row is a state - the board state after a move from its
        # respective plot (e.g. the bottom row is the state after moving from plot #5).
        bank_gains = possible_states[:, 6] - board[6]  # How much was added to the bank
        possible_rewards = -1 * np.ones(6)  # Initialization
        possible_rewards[row_before == 0] = self.invalid_reward  # Set reward for illegal moves
        possible_rewards[(possible_rewards == -1) & hit_bank] = 2  # Set reward for hitting the bank. TODO: Use move_again?
        possible_rewards[possible_rewards == -1] = bank_gains[possible_rewards == -1]  # All the rest
        return possible_states[:, :-1], possible_rewards

    def _greedy_action(self, board):
        """
        Determines the action with the highest Q. Returns both the move and its reward.
        """
        possible_states, possible_rewards = self._reward(board)
        q_options = possible_rewards + self.gamma * tf.reduce_max(self.q_network(possible_states), axis=1)
        decision = int(tf.argmax(q_options))
        return decision, possible_rewards[decision]

    def update_epsilon(self):
        self.epsilon = max(self.e_min, self.epsilon * self.e_decay)

    def set_weights(self, w_lst: list[np.ndarray], b_lst: list[np.ndarray]):
        """
        Sets the weights of the player's networks.
        The order of the lists corresponds to the order of the layers, from the input layer to the output.
        :param w_lst: A list of the W matrices.
        :param b_lst: A list of the b vectors.
        """
        for i in range(len(w_lst)):
            self.q_network.layers[i].set_weights([w_lst[i], b_lst[i]])
            self.target_q_network.layers[i].set_weights([w_lst[i], b_lst[i]])

    @staticmethod
    def flip_board(board):
        return np.roll(board, 7)


if __name__ == '__main__':
    p = ReinforcementPlayer(5)
    b = gameplay.gen_new_board()
    b[4] = 0

