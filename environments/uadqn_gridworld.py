"""
Gridworld for UADQN
"""
from random import uniform
import numpy as np
import torch
import os
import json
from environments.gridworld import Gridworld

class UADQNGridworld(Gridworld):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        class action_space():
            def __init__(self):
                self.n = 4

        class observation_space():
            def __init__(self_):
                self_.shape = [self.width*self.height]

        self.action_space = action_space()
        self.observation_space = observation_space()
        self.num_states = self.width*self.height

        self.reset()

    def reset(self):
        self.state = np.random.choice(np.arange(self.width*self.height, self.start_distribution))
        return self.return_obs()

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
        self.state = next_state

        return self.return_obs(), reward, done, info

    def reset(self):
        """
        Resets the environment and returns the first observation
        """
        self.state = np.random.choice(np.arange(self.num_states), p=self.start_distribution)
        self.timestep = 0
        self.disc_return = 0
        return self.return_obs()

    def return_obs(self):
        """
        Returns state
        """
        self.encoded_state = np.zeros(self.num_states)
        self.encoded_state[self.state] = 1
        obs = np.array(self.encoded_state)
        return torch.tensor(obs).float()

    def seed(self, seed):
        """
        Set random seed of the environment
        """

    def render(self):
        """
        Can be used to optionally render the environment
        """

    def save(self, file_path):
        init_arg_dict = {
            'width':self.width,
            'height':self.height,
            'start_position':self.start_position,
            'goal_positions':self.goal_positions,
            'cliff_positions':self.cliff_positions,
            'success_reward':self.success_reward,
            'failure_reward':self.failure_reward,
            'step_reward':self.step_reward,
            'p_successful_step':self.p_successful_step,
            'random_step_type':self.random_step_type,
            'gamma':self.gamma,
            'max_timesteps':self.max_timesteps,
            'start_distribution':list(self.start_distribution)
        }
        transition_matrix = list(self.transition_matrix.flatten())
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(os.path.join(file_path,'gridworld_init_dict'), 'w+') as f:
            json.dump(init_arg_dict, f)
        with open(os.path.join(file_path,'mlp_transition'), 'w+') as f:
            json.dump(transition_matrix, f)
    
    def load(self, init_args, transition_matrix=None):
        self.__init__(**init_args)
        if transition_matrix is not None:
            self.transition_matrix = np.reshape(np.array(transition_matrix), (self.num_states, self.num_actions, self.num_states))

if __name__ == '__main__':

    from utils.value_iteration import VIMDPSolver

    gridworld = Gridworld(width=3,
        height=3,
        start_position=(0,0),
        goal_positions=[(2,2)],
        cliff_positions = [(0,2),(1,2)],
        success_reward=1,
        failure_reward=0,
        step_reward=0,
        p_successful_step=.5,
        gamma=.999,
        max_timesteps=1000)

    solver = VIMDPSolver(1e-3)
    
    policy, values = solver.solve_mdp(gridworld)
    print(policy, values)