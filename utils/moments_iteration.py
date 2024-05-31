import numpy as np
from environments.mdp import TabularMDP
from utils.moments import MomentsCalculator
from utils.value_iteration import VIMDPSolver

class MomentsVI:

    def __init__(self, epsilon):

        self.epsilon = epsilon #value iteration tolerance parameter
    
    def solve_mdp(self, mdp: TabularMDP, alpha=0):

        V = np.zeros(mdp.num_states)
        M = np.zeros(mdp.num_states)

        policy = np.zeros(mdp.num_states)
        delta = np.inf

        while delta > self.epsilon:
            
            Q_next = (mdp.transition_matrix*(mdp.reward_matrix+mdp.gamma*V.reshape(1,1,-1))).sum(2)
            EZ2_sa = (mdp.transition_matrix*(mdp.reward_matrix**2 + 2*mdp.gamma*(V.reshape(1,1,-1))*mdp.reward_matrix 
                                             + (mdp.gamma**2)*M.reshape(1,1,-1))).sum(2)
            std = np.sqrt(EZ2_sa-Q_next**2)
            policy = np.argmax(Q_next-alpha*std, 1)

            V_next = np.take_along_axis(Q_next, policy.reshape(-1,1), 1)
            M_next = np.take_along_axis(EZ2_sa, policy.reshape(-1,1), 1)

            delta_V = np.abs(V_next - V).max()
            delta_M = np.abs(M_next - M).max()

            V = V_next
            M = M_next

            delta = max(delta_V, delta_M)

        return policy, V, M
