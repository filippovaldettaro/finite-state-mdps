import numpy as np
from environments.mdp import TabularMDP

class VIMDPSolver:

    def __init__(self, epsilon):

        self.epsilon = epsilon #value iteration tolerance parameter
    
    def solve_mdp(self, mdp: TabularMDP):

        V = np.zeros(mdp.num_states)
        delta = np.inf

        while delta > self.epsilon:
            
            Q_next = (mdp.transition_matrix*(mdp.reward_matrix+mdp.gamma*V.reshape(1,1,-1))).sum(2)
            V_next = Q_next.max(1)

            delta = np.abs(V_next - V).max()

            V = V_next
        
        policy = np.zeros(mdp.num_states, dtype=int)
        
        for state in range(mdp.num_states):
            policy[state] = Q_next[state].argmax()

        return policy, V