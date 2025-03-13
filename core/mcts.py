import logging
import math
from typing import Dict, List, Tuple, Final, TypeAlias
from numpy.typing import NDArray

import numpy as np

from core.config import MCTSConfig, LOGGER_NAME
from core.interfaces import IGame, INeuralNetWrapper

# Constants
EPS: Final[float] = 1e-8  # Small constant to prevent division by zero

# Type aliases for improved readability
StateStr: TypeAlias = str  # String representation of a game state
Action: TypeAlias = int  # Integer index into the action space
StateAction: TypeAlias = Tuple[StateStr, Action]  # (state, action) pair
PolicyVector: TypeAlias = NDArray[np.float64]  # Probability distribution over actions
ValidMoves: TypeAlias = NDArray[np.bool_]  # Binary mask of legal moves

log = logging.getLogger(LOGGER_NAME)


class MCTS:
    """
    Monte Carlo Tree Search implementation for game playing.
    
    This class implements the MCTS algorithm with neural network guidance,
    following the AlphaZero approach. The search is guided by:
    1. Prior probabilities from a policy network
    2. Value estimates from a value network
    3. Visit counts for exploration/exploitation balance
    
    The search tree is implicitly stored through dictionaries mapping:
    - States to their neural network policy predictions
    - State-action pairs to their Q-values and visit counts
    - States to their total visit counts
    
    The tree is built incrementally through repeated simulations, each consisting of:
    1. Selection: Choose actions recursively using UCB formula
    2. Expansion: Add a new leaf node to the tree
    3. Evaluation: Use neural network to evaluate the leaf
    4. Backpropagation: Update statistics for all visited nodes
    """

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, config: MCTSConfig) -> None:
        """
        Initialize the MCTS with game rules and neural network.

        Args:
            game: Game implementation providing rules and mechanics
            nnet: Neural network for policy and value predictions
            config: Configuration parameters for MCTS
        """
        self.game = game
        self.nnet = nnet
        self.config = config
        
        # Tree statistics
        self.Qsa: Dict[StateAction, float] = {}  # Q values for state-action pairs
        self.Nsa: Dict[StateAction, int] = {}  # Visit counts for state-action pairs
        self.Ns: Dict[StateStr, int] = {}  # Visit counts for states
        self.Ps: Dict[StateStr, PolicyVector] = {}  # Initial policies for states
        self.Es: Dict[StateStr, float] = {}  # Game-ended status for states
        self.Vs: Dict[StateStr, ValidMoves] = {}  # Valid moves mask for states

    def get_action_prob(self, canonical_board: NDArray, temp: float = 1) -> List[float]:
        """
        Get action probabilities for the current board state.

        Performs multiple MCTS simulations and returns a probability distribution
        over possible actions based on the visit counts of each action.

        Args:
            canonical_board: The game board in canonical form
            temp: Temperature parameter controlling exploration:
                 - temp=0: Choose the best action deterministically
                 - temp=1: Choose proportionally to visit counts
                 - temp>1: Increase exploration
                 - 0<temp<1: Reduce exploration

        Returns:
            List[float]: Probability distribution over all possible actions.
                        The probability of illegal moves will be zero.
        """
        # Perform simulations to build the search tree
        for _ in range(self.config.num_mcts_sims):
            self.search(canonical_board)

        # Extract visit counts for all actions
        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        # Handle temperature=0 case (deterministic best action)
        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        # Apply temperature and normalise to get probabilities
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        return [x / counts_sum for x in counts]

    def search(self, canonical_board: NDArray) -> float:
        """
        Perform one iteration of MCTS.

        This function implements the four phases of MCTS:
        1. Selection: Choose actions recursively according to UCB formula
        2. Expansion: Add a new leaf node to the tree
        3. Evaluation: Use neural network to evaluate the leaf
        4. Backpropagation: Update statistics for all visited nodes

        Args:
            canonical_board: The game board in canonical form

        Returns:
            float: The negative of the evaluated position value (for opponent's perspective)
                  Values are in the range [-1, 1] where:
                  - 1 means the current player is winning
                  - -1 means the current player is losing
                  - 0 means the position is equal

        Notes:
            - The return value is negated because the value is from the opponent's perspective
            - TODO: Consider optimising the action iteration for games with sparse legal moves
        """
        s = self.game.string_representation(canonical_board)

        # PHASE 1: Check if game has ended
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)  # TODO: Player always White
        if self.Es[s] != 0:
            return -self.Es[s]

        # PHASE 2: Leaf node - evaluate position with neural network
        if s not in self.Ps:
            # Get policy and value predictions
            self.Ps[s], v = self.nnet.predict(canonical_board)
            valids = self.game.valid_move_masking(canonical_board, 1)  # TODO: Player always White
            
            # Mask invalid moves and renormalise probabilities
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # All valid moves were masked - use uniform distribution over valid moves
                # This can indicate issues with neural network architecture or training
                log.error("All valid moves were masked, using uniform distribution.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # Initialize node statistics
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # PHASE 3: Choose action according to UCB formula
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Pick the action with the highest upper confidence bound
        # TODO: This is inefficient for games with sparse legal moves (e.g. BlokusDuo)
        #       Consider iterating only over valid moves instead
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = (self.Qsa[(s, a)] + 
                         self.config.cpuct * self.Ps[s][a] * 
                         math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)]))
                else:
                    u = (self.config.cpuct * self.Ps[s][a] * 
                         math.sqrt(self.Ns[s] + EPS))

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # PHASE 4: Recursively evaluate chosen action
        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # Update node statistics
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = ((self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / 
                               (self.Nsa[(s, a)] + 1))
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
