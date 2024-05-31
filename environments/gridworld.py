"""
Generic tabular MDP
"""
from random import uniform
import numpy as np
import os
import json
from environments.mdp import TabularMDP

class Gridworld(TabularMDP):
    
    def __init__(self,
                width=None,
                height=None,
                start_position=None,
                goal_positions=[],
                cliff_positions = [],
                success_reward=1,
                failure_reward=0,
                step_reward=0,
                p_successful_step=1,
                random_step_type=None, # 'directional', 'uniform', 'wind'
                gamma=1,
                max_timesteps=None,
                start_distribution=None
                ):


        super().__init__(num_states=width*height,
                num_actions=4,
                transition_matrix=None,
                reward_matrix=None,
                start_distribution=start_distribution,
                gamma=gamma,
                terminal_states=None)

        self.width = width
        self.height = height
        self.start_position = start_position
        self.start_distribution=start_distribution
        if self.start_distribution is not None and self.start_distribution!='uniform':
            self.start_distribution = np.array(start_distribution)
        if start_position is not None:
            self.start_distribution = np.zeros(self.num_states)
            start_state = self.get_state(start_position)
            self.start_distribution[start_state] = 1
        self.goal_positions = goal_positions
        self.cliff_positions = cliff_positions
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.step_reward = step_reward
        self.p_successful_step = p_successful_step

        self.goal_states = tuple([self.get_state(goal_pos) for goal_pos in goal_positions])
        self.cliff_states = tuple([self.get_state(cliff_pos) for cliff_pos in cliff_positions])
        self.terminal_states = tuple(list(self.goal_states) + list(self.cliff_states))
        self.max_timesteps = max_timesteps

        if start_distribution == 'uniform':
            uniform_p = 1/(self.num_states-len(self.terminal_states))
            self.start_distribution = np.ones(self.num_states) * uniform_p
            for terminal in self.terminal_states:
                self.start_distribution[terminal] = 0

        self.random_step_type = random_step_type
        if random_step_type is None:
            self.random_step_type = 'directional'   #impossible to go backwards, could be nudged to either side
        
        self.transition_matrix = self.get_transition_matrix()
        self.reward_matrix = self.get_reward_matrix()

    
    def get_coords(self, state):
        return (state%self.width, int(state/self.width))

    def get_state(self, coords):
        x, y = coords
        return self.width*y+x

    def get_neighbour_states(self, state):
        #return (up, down, left, right) states with itself in the tuple if on the edge
        x, y = self.get_coords(state)
        up = self.get_state((x, max(y-1, 0)))
        right = self.get_state((min(x+1, self.width-1), y))
        down = self.get_state((x, min(y+1, self.height-1)))
        left = self.get_state((max(x-1, 0), y))

        return (up, right, down, left)

    def get_transition_matrix(self):
        transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
        if self.random_step_type == 'directional':
            for state in range(self.num_states):
                neighbours = self.get_neighbour_states(state)
                for action in range(self.num_actions):
                    front = neighbours[action]
                    side_right = neighbours[(action+1)%self.num_actions]
                    side_left = neighbours[(action-1)%self.num_actions]
                    behind = neighbours[(action+2)%self.num_actions]

                    transition_matrix[state, action, front] += self.p_successful_step
                    transition_matrix[state, action, side_right] += (1-self.p_successful_step)/2
                    transition_matrix[state, action, side_left] += (1-self.p_successful_step)/2
                    
        elif self.random_step_type == 'uniform':
            for state in range(self.num_states):
                neighbours = self.get_neighbour_states(state)
                for action in range(self.num_actions):
                    front = neighbours[action]
                    side_right = neighbours[(action+1)%self.num_actions]
                    side_left = neighbours[(action-1)%self.num_actions]
                    behind = neighbours[(action+2)%self.num_actions]

                    transition_matrix[state, action, front] += self.p_successful_step
                    transition_matrix[state, action, side_right] += (1-self.p_successful_step)/3
                    transition_matrix[state, action, side_left] += (1-self.p_successful_step)/3
                    transition_matrix[state, action, behind] += (1-self.p_successful_step)/3
         
        elif self.random_step_type == 'wind':
            for state in range(self.num_states):
                neighbours = self.get_neighbour_states(state)
                for action in range(self.num_actions):
                    front = neighbours[action]
                    beneath = neighbours[2]

                    transition_matrix[state, action, front] += self.p_successful_step
                    transition_matrix[state, action, beneath] += 1-self.p_successful_step
                
        for state in self.terminal_states:
            transition_matrix[state] = 0.
            transition_matrix[state, :, state] = 1.
        
        for state in range(self.num_states):
            for action in range(self.num_actions):
                assert transition_matrix[state, action].sum() == 1.
        
        return transition_matrix
                
    def get_reward_matrix(self):
        reward_matrix = self.step_reward*np.ones((self.num_states, self.num_actions, self.num_states))
        for goal_state in self.goal_states:
            reward_matrix[:,:,goal_state] = self.success_reward
        for cliff_state in self.cliff_states:
            reward_matrix[:,:,cliff_state] = self.failure_reward
        for state in self.terminal_states:
            reward_matrix[state,:,state] = 0.
        
        return reward_matrix

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