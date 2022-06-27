import time

import gym
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from network import FeedForwardNN


class PPO:
    def __init__(self, env, **hyperparameters):
        """
        Initializes the PPO model, including hyperparameters.
        Parameters:
            policy_class - the policy class to use for our actor/critic networks.
            env - the environment to train on.
            hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
        Returns:
            None
        """

        # Make sure the environment is compatible with our code
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Box

        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # ALG STEP 1
        # Initialize actor and critic networks
        # initial policy parameters \theta_0, initial value function parameters \phi_0
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # initialize actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        # Miscellaneous parameters
        self.render = False  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        # Sets the seed of our program, used for reproducibility of results
        self.seed = None

    def _init_hyperparameters(self):
        """
        Initialize default and custom values for hyperparameters
        Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                                        hyperparameters defined below with custom values.
        Return:
                None
        """

        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800  # timesteps per batch
        self.max_timesteps_per_episode = 1600  # timesteps per episode

        # Number of times to update actor/critic per iteration
        self.n_updates_per_iteration = 5

        # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.clip = 0.2
        # Learning rate of actor optimizer
        self.lr = 0.005
        # Discount factor to be applied when calculating Rewards-To-Go
        self.gamma = 0.95

    def get_action(self, obs):
        """
        Queries an action from the actor network, should be called from rollout.
        Parameters:
                obs - the observation at the current timestep
        Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """

        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
        Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """

        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self):
        """
        Too many transformers references, I'm sorry. This is where we collect the batch of data
        from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
        of data each time we iterate the actor/critic networks.
        Parameters:
                None
        Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data
        batch_obs = (
            []
        )  # observations: (number of timesteps per batch, dimension of observation)
        batch_acts = []  # actions: (number of timesteps per batch, dimension of action)
        batch_log_probs = []  # log probabilities: (number of timesteps per batch)
        batch_rews = (
            []
        )  # rewards: (number of episodes, number of timesteps per episode)
        batch_rtgs = []  # reward-to-go’s: (number of timesteps per batch)
        batch_lens = []  # batch lengths: (number of episodes)

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and len(batch_lens) == 0:
                    self.env.render()

                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.
        Parameters:
                total_timesteps - the total number of timesteps to train for
        Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:  # ALG STEP 2
            # Increment t_so_far somewhere below
            # ALG STEP 3
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
            ) = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # calculate V_{phi, k} and pi_theta(a_t | s_t)
            V, _ = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)

            # ALG STEP 5
            # calculate advantage
            A_k = batch_rtgs - V.detach()

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Save our model if it's time
                if i_so_far % self.save_freq == 0:
                    torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                    torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.
        Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of action)
        Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """

        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs


if __name__ == "__main__":

    env = gym.make("Pendulum-v1")
    model = PPO(env)
    model.learn(10000)
