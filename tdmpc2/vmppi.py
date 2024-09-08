import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class VMPPI:
	"""
	Value-MPPI agent (~TD-MPC2 without latent model). Implements training + inference.
	Can be used for single-task experiments,
	supports point cloud observations.
	"""

	def __init__(self, cfg, env):
		self.cfg = cfg
		self.envs = env
		self.device = torch.device(f'cuda:{cfg.gpu}')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._Qs.parameters()},
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.discount = self._get_discount(cfg.episode_length)

		self.gpu_act = cfg.gpu_act and torch.cuda.is_available()

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the environment.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True)
		z = self.model.encode(obs)
		if self.cfg.mpc:
			action = self.plan(z, t0=t0, eval_mode=eval_mode)
		# else:
		# 	action = self.model.pi(z)[int(not eval_mode)]
		return action.cpu() if not self.gpu_act else action

	@torch.no_grad()
	def _estimate_value(self, actions):
		"""Estimate value of a trajectory starting at current state and executing given actions."""
		value = torch.zeros(self.cfg.num_samples, device=self.device)
		start_id = 0
		end_id = self.cfg.num_envs

		while start_id < self.cfg.num_samples:
			action_batch = actions[:, start_id:end_id]
			G, discount, obs, terminated = 0, 1, None, 0
			
			for t in range(self.cfg.horizon):

				obs, reward, done, _ = self.envs.step(action_batch[t], get_obs=(t == self.cfg.horizon - 1))

				if t == self.cfg.horizon - 1:
					z = self.model.encode(obs)

				G += discount * (1 - terminated) * reward
				discount *= self.discount
				terminated = torch.clip_(terminated + done.float(), max=1.)
			
			z = self.model.encode(obs)
			if self.cfg.use_policy_prior:
				# Use the policy to estimate the value of the terminal state
				a_h = self.model.pi(z)[1]
			else:
				# Use MPPI to estimate the value of the terminal state
				a_h = self._estimate_next_action(z)
			value[start_id:end_id] = G + discount * (1 - terminated) * self.model.Q(z, a_h, return_type='avg').squeeze()

			start_id = end_id
			end_id = min(end_id + self.cfg.num_envs, self.cfg.num_samples)

		return value
	
	@torch.no_grad()
	def _estimate_next_action(self, z, k=3, num_samples=256, n_elites=32):
		"""Estimate a Q function at a certain latent state by sampling H=1 actions and performing k iterations of MPPI updates"""
		mean = torch.zeros(1, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(1, self.cfg.action_dim, device=self.device)

		value_estimator = lambda actions: self.model.Q(z, actions[0], return_type='avg', target=True)

		actions, mean, std = self._iterate_mppi(mean, std, 1, k, num_samples, n_elites, value_estimator)

		return actions[0]

	@torch.no_grad()
	def _iterate_mppi(self, mean, std, horizon, iterations, num_samples, n_elites, value_estimator):
		 
		for _ in range(iterations):
			# Sample actions			
			actions = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(horizon, num_samples, self.cfg.action_dim, device=std.device)).clamp(-1, 1)
			
			# Compute elite actions
			value = value_estimator(actions).nan_to_num_(0)		
			elite_idxs = torch.topk(value, n_elites).indices
			elite_value = torch.gather(value, 0, elite_idxs)
			elite_actions = torch.gather(actions, 1, elite_idxs.unsqueeze(0).unsqueeze(2).expand(horizon, -1, self.cfg.action_dim))
			
			# Update parameters
			max_value = elite_value.max()
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= (score.sum(0) + 1e-9)
			mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1)
			std = torch.sqrt(torch.sum(score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
		
		# Select action sequence with probability `score`
		score = score.squeeze().cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]

		return actions, mean, std

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False):
		"""
		Plan a sequence of actions using the environment
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		if self.cfg.use_policy_prior and self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(0).repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z)[1]
				_z = self.model.next(_z, pi_actions[t])
			pi_actions[-1] = self.model.pi(_z)[1]

		self.envs.planning_mode()
		
		# Initialize state and parameters
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
	
		if self.cfg.use_policy_prior and self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
			n_samples = self.cfg.num_samples - self.cfg.num_pi_trajs
		else:
			n_samples = self.cfg.num_samples
	
		# Iterate MPPI
		mppi_actions, mean, std = self._iterate_mppi(mean, std, self.cfg.horizon, self.cfg.iterations, n_samples, self.cfg.num_elites, self._estimate_value)

		if self.cfg.use_policy_prior and self.cfg.num_pi_trajs > 0:
			actions[:, self.cfg.num_pi_trajs:] = mppi_actions
		else:
			actions = mppi_actions

		self._prev_mean = mean
		action, std = actions[0], std[0]
		if not eval_mode:
			action += std * torch.randn(self.cfg.action_dim, device=std.device)

		self.envs.real_mode()

		return action.clamp_(-1, 1)
		
	def update_pi(self, zs):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs)
		qs = self.model.Q(zs, pis, return_type='avg')
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		if self.cfg.use_policy_prior:
			pi = self.model.pi(next_z)[1]
		else:
			pi = self._estimate_next_action(next_z)
		discount = self.discount
		return reward + discount * (1 - terminated) * self.model.Q(next_z, pi, return_type='min', target=True)

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, _ = buffer.sample()

		zs = self.model.encode(obs)

		# Compute targets
		with torch.no_grad():
			td_targets = self._td_target(zs[1:], reward, terminated)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, return_type='all')
		
		# Compute losses
		value_loss = 0
		for t in range(self.cfg.horizon):
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		ret_dict = {
			"value_loss": float(value_loss.mean().item()),
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
		}

		# Update policy, if necessary
		if self.cfg.use_policy_prior:
			pi_loss = self.update_pi(zs.detach())
			ret_dict["pi_loss"] = pi_loss
			ret_dict["pi_scale"] = float(self.scale.value)
		
		# Update target Q-functions
		self.model.soft_update_target_Q()

		self.model.eval()

		# Return training statistics		
		return ret_dict
