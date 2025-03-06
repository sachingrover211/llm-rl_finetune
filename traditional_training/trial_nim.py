import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NimEnv(gym.Env):
    """
    A two-player Nim environment.
    
    Observation:
        Type: Box(shape=(num_piles,), low=0, high=max_possible_stones)
        The number of stones in each pile.

    Action:
        Type: MultiDiscrete([num_piles, max_remove+1])
        - The first component is which pile to take from (0 to num_piles-1).
        - The second component is how many stones to remove (1 to max_remove).
          (An action of (pile_index, 0) is invalid and will be ignored.)
          Typically, you only want to remove up to the number of stones in the chosen pile.
          If the chosen removal is invalid (e.g., removing more stones than are in the pile),
          this step will be considered illegal but for simplicity, we clamp the removal
          to be valid (or skip if none is valid).

    Reward:
        +1 if the agent wins (takes the last stone),
        -1 if the opponent wins,
         0 otherwise (intermediate steps).

    Episode Termination:
        1. All piles are empty (someone has taken the last stone).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        num_piles=3,
        init_stones=(3, 5, 7),
        max_remove=None,
        opponent="random",
        seed=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_piles = num_piles
        self.init_stones = np.array(init_stones, dtype=np.int32)

        # If max_remove is None, allow removing up to the largest pile size
        self.max_stones = int(max(self.init_stones))
        self.max_remove = max_remove if max_remove is not None else self.max_stones

        self.opponent = opponent

        # Define the observation space (number of stones in each pile)
        # We'll allow up to `self.max_stones` in each pile for simplicity
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_stones,
            shape=(self.num_piles,),
            dtype=np.int32
        )

        # Define the action space:
        # (which pile to take from, how many stones to remove)
        self.action_space = spaces.MultiDiscrete([self.num_piles, self.max_remove + 1])

        self.np_random = None
        self.seed(seed)

        # Internal state
        self.state = None       # array of shape (num_piles,)
        self.current_player = 0 # 0 = agent, 1 = opponent

        self.reset()

    def seed(self, seed=None):
        """Sets the seed for this environment's random number generator(s)."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to an initial state and returns
        the initial observation.
        """
        if seed is not None:
            self.seed(seed)

        # Reset piles to the predefined init_stones.
        self.state = self.init_stones.copy()
        # Agent always starts (player 0)
        self.current_player = 0

        if self.render_mode == "human":
            self.render()

        # Return initial observation and an info dict
        return self._get_obs(), {}

    def step(self, action):
        """
        The agent takes a step in the environment (removes stones).
        Then, if the game is not done, the environment's opponent also makes a move.
        """
        reward = 0.0
        done = False
        info = {}

        # ----- Agent's move -----
        self._take_action(action, player=0)

        # Check if agent ended the game
        if self._all_piles_empty():
            # Agent took the last stone
            reward = 1.0
            done = True
        else:
            # ----- Opponent's move -----
            self._opponent_move()

            # Check if opponent ended the game
            if self._all_piles_empty():
                # Opponent took the last stone
                reward = -1.0
                done = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, done, False, info

    def render(self):
        """Print the state of the piles to the terminal."""
        print(f"Current piles: {self.state}")

    def _get_obs(self):
        """Return the current state as the observation."""
        return self.state.copy()

    def _all_piles_empty(self):
        """Check if all piles are empty."""
        return np.all(self.state == 0)

    def _take_action(self, action, player):
        """
        Execute a given action for the specified player.
        Action is (pile_index, stones_to_remove).
        If invalid, we try to fix or skip it (here we clamp).
        """
        pile_index, stones_to_remove = action

        # We must remove at least 1 stone
        stones_to_remove = max(1, stones_to_remove)

        if pile_index < 0 or pile_index >= self.num_piles:
            return  # Invalid pile index, skip

        # Clamp removal if it's too large
        stones_to_remove = min(stones_to_remove, self.state[pile_index])
        self.state[pile_index] -= stones_to_remove

    def _opponent_move(self):
        """
        A simple opponent policy.
        By default, "random": choose a random valid move.
        """
        if self.opponent == "random":
            # Collect all valid moves
            valid_moves = []
            for p_idx in range(self.num_piles):
                
                # For each pile, can remove from 1..pile_size stones
                pile_size = self.state[p_idx]
                for r in range(1, pile_size + 1):
                    valid_moves.append((p_idx, r))

            if len(valid_moves) > 0:
                move = self.np_random.choice(len(valid_moves))
                chosen_move = valid_moves[move]
                self._take_action(chosen_move, player=1)
        else:
            # You can implement a smarter (e.g., optimal) opponent here.
            self._take_optimal_move()

    def _take_optimal_move(self):
        """
        Example of an optimal Nim strategy move, if you want the environment to play optimally.
        This is optional. By default, we do "random".
        """
        # Calculate Nim-sum
        nim_sum = 0
        for pile_count in self.state:
            nim_sum ^= pile_count

        if nim_sum == 0:
            # No winning move; pick a random move as fallback
            self._opponent_move()
        else:
            # Find a pile to reduce
            for p_idx in range(self.num_piles):
                pile_count = self.state[p_idx]
                target = pile_count ^ nim_sum
                if target < pile_count:
                    # Make this move
                    stones_to_remove = pile_count - target
                    self._take_action((p_idx, stones_to_remove), player=1)
                    return



if __name__ == "__main__":
    env = NimEnv(render_mode="human", num_piles=1, init_stones=(11,), opponent="random")
    obs, info = env.reset()
    done = False

    while not done:
        # Example random agent:
        action = env.action_space.sample()
        print(action)
        obs, reward, done, truncated, info = env.step(action)
        print('action:', action)
        # print('obs:', obs)
        
        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("Opponent wins!")
