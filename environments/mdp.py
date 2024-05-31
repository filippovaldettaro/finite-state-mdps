"""
Generic tabular MDP
"""
import numpy as np
from random import randint

class TabularMDP():
    
    def __init__(self,
                 num_states,
                 num_actions,
                 transition_matrix,
                 reward_matrix,
                 start_distribution=None,
                 gamma = 1.,
                 terminal_states = [],
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.start_distribution = start_distribution
        self.terminal_states = terminal_states

        self.transition_matrix = transition_matrix  #[S,A,S']           # T or R are "None", taken as placeholder mdp with unknown transition/rewards
        self.reward_matrix = reward_matrix          #[S,A,S']

    def step(self, action):
        """
        Requires: selected action
        Returns: observation, reward, done (boolean), and optional Notes
        """
        state = self.state
        transition_probabilities = self.transition_matrix[state, action]
        
        next_state = np.random.choice(np.arange(self.num_states), p=transition_probabilities)
        reward = self.reward_matrix[state, action, next_state]

        self.timestep += 1
        
        done = False
        if next_state in self.terminal_states:
            done = True
        if self.timestep == self.max_timesteps:
            done = True

        info = None

        return self.return_obs(), reward, done, info

    def reset(self):
        """
        Resets the environment and returns the first observation
        """
        self.state = np.random.choice(np.arange(self.num_states), p=self.start_distribution)
        self.timestep = 0
        self.disc_return = 0
        return self.return_obs()

    def get_neighbour_states(self, state):
        return tuple(range(self.num_states))

    def return_obs(self):
        """
        Returns state
        """
        obs = np.array(self.state)
        return obs

    def seed(self, seed):
        """
        Set random seed of the environment
        """

    def render(self):
        """
        Can be used to optionally render the environment
        """
