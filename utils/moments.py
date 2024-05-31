import numpy as np
from environments.mdp import TabularMDP
from utils.value_iteration import VIMDPSolver

class MomentsCalculator:
    def __init__(self, mdp: TabularMDP, policy: np.array):
        self.mdp = mdp
        self.policy = policy
        self.num_states = mdp.num_states
        self.gamma = mdp.gamma
        if len(policy.shape)==1:
            policy_indexing = np.expand_dims(self.policy, (1,2))    
            self.transition_matrix = np.take_along_axis(self.mdp.transition_matrix, policy_indexing, 1).squeeze() #shape [s,s']
            self.reward_matrix = np.take_along_axis(self.mdp.reward_matrix, policy_indexing, 1).squeeze()         #shape [s,s']

        elif len(policy.shape)==2:
            policy = np.expand_dims(self.policy, (2,))    
            self.transition_matrix = np.sum(policy*self.mdp.transition_matrix, 1).squeeze()   #shape [s,s']
            self.reward_matrix = np.sum(policy*self.mdp.reward_matrix, 1).squeeze()       #shape [s,s']

        if self.gamma == 1:
            to_invert = np.identity(self.num_states) - self.transition_matrix
            self.inverse = np.linalg.inv(to_invert)
        
        self.means = None
        
    def get_expected_value(self):
        
        to_invert = np.identity(self.num_states) - self.mdp.gamma*self.transition_matrix
        inverse = np.linalg.inv(to_invert)

        self.means = inverse@((self.transition_matrix*self.reward_matrix).sum(1))
        return self.means
    
    def get_variance(self):
        if self.means is None:
            self.means = self.get_expected_value()
        
        to_invert = np.identity(self.num_states) - (self.mdp.gamma**2)*self.transition_matrix
        inverse = np.linalg.inv(to_invert)
        known_quantity = self.transition_matrix*(self.reward_matrix**2 + 2*self.mdp.gamma*self.reward_matrix*np.expand_dims(self.means, 0))
        self.second_moments = inverse@(known_quantity.sum(1))
        return self.second_moments - self.means**2