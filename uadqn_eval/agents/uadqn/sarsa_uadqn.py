from uadqn_eval.agents.uadqn.uadqn import UADQN
import torch
import torch.nn.functional as F

class SARSA_UADQN(UADQN):

    def __init__(self, policy_arr, **kwargs):
        self.policy_arr = torch.tensor(policy_arr)
        super().__init__(**kwargs)

    def policy(self, state):
        state = torch.argmax(state, dim=-1)
        return self.policy_arr[state] #also works for batched states

    def act(self, state):
        """
        Act according to the policy being evaluated
        """

        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)

        posterior1 = self.posterior1(state).view(self.env.action_space.n, self.n_quantiles)
        posterior2 = self.posterior2(state).view(self.env.action_space.n, self.n_quantiles)

        action = self.policy(state)

        # Calculate aleatoric uncertainty

        biased_uncertainties_aleatoric = torch.sqrt(torch.std(net, dim=1)[action])

        covariance = torch.mean((posterior1-torch.mean(posterior1,dim=1).unsqueeze(1))*(posterior2-torch.mean(posterior2,dim=1).unsqueeze(1)), dim=1) #fixed dimension in mean
        uncertainties_aleatoric = torch.sqrt(F.relu(covariance)[action])

        # Calculate epistemic uncertainty
        uncertainties_epistemic = torch.sqrt(torch.mean((posterior1-posterior2)**2, dim=1)/2)[action]

        #print(mean_action_values, torch.sqrt(uncertainties_epistemic), torch.sqrt(uncertainties_aleatoric))
        
        if self.logging: #and self.this_episode_time == 0:
            self.logger.add_scalar('State', state.argmax(), self.timestep)
            self.logger.add_scalar('Biased Aleatoric Uncertainty', biased_uncertainties_aleatoric, self.timestep)
            self.logger.add_scalar('Epistemic Uncertainty', uncertainties_epistemic, self.timestep)
            self.logger.add_scalar('Aleatoric Uncertainty', uncertainties_aleatoric, self.timestep)
        
        return self.policy(state)
    
    def train_step(self, transitions):
        """
        Performs gradient descent step on a batch of transitions
        """
        states, actions, rewards, states_next, dones = transitions

        #debug state shapes...

        # Calculate target Q
        with torch.no_grad():
            target = self.target_network(states_next.float())
            target = target.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)

        # Calculate SARSA target Q values
        next_action_idx = self.policy(states_next).unsqueeze(-1).unsqueeze(-1).to(self.device)
        q_value_target = target.gather(1, next_action_idx.repeat(1, 1, self.n_quantiles))

        # Calculate TD target
        rewards = rewards.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        dones = dones.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value of actions played
        outputs = self.network(states.float())
        outputs = outputs.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        actions = actions.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        q_value = outputs.gather(1, actions)

        # TD loss for main network
        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Calculate predictions of posterior networks
        posterior1 = self.posterior1(states.float())
        posterior1 = posterior1.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior1 = posterior1.gather(1, actions)

        posterior2 = self.posterior2(states.float())
        posterior2 = posterior2.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior2 = posterior2.gather(1, actions)

        # Regression loss for the posterior networks
        loss_posterior1 = self.loss(posterior1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss_posterior2 = self.loss(posterior2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss += loss_posterior1 + loss_posterior2

        # Anchor loss for the posterior networks
        anchor_loss = self.calc_anchor_loss()        
        loss += anchor_loss

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), anchor_loss.mean().item()