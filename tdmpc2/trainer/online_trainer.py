from time import time

import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		# Distribute the evaluation episodes evenly across the environments so that not only the shortest runs are collected
		eps_to_collect = torch.zeros(self.cfg.num_envs, dtype=torch.int32)
		ees = self.cfg.eval_episodes
		while ees > 0:
			eps_to_collect += (ees > torch.arange(self.cfg.num_envs)).int()
			ees -= self.cfg.num_envs

		ep_rewards = []
		obs, done, ep_reward, t = (self.env.reset(), torch.tensor(False), torch.zeros(self.cfg.num_envs, device=self.env.device),
					  				torch.zeros(self.cfg.num_envs, dtype=torch.int32))
		if self.cfg.save_video:
			self.logger.video.init(self.env, enabled=True)
		while eps_to_collect.sum() > 0:
			action = self.agent.act(obs, t0=t==0, eval_mode=True)
			obs, reward, done, info = self.env.step(action)
			ep_reward += reward
			t += 1
			if self.cfg.save_video and not done[0]:  # TODO: this assumes env[0] is the only one collecting video
				self.logger.video.record(self.env)
			
			for idx, d in enumerate(done):
				if d and eps_to_collect[idx]:
					eps_to_collect[idx] -= 1
					ep_rewards.append(ep_reward[idx].item())
					ep_reward[idx] = 0.
					t[idx] = 0

					if self.cfg.save_video and not idx:
						# FIXME: this is a hack to save the video for the first environment only
						self.logger.video.save(self._step)
		return dict(
			episode_reward=sum(ep_rewards) / len(ep_rewards),
			# episode_success=info['success'].mean(),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device=self.env.device)
		else:
			obs = obs.unsqueeze(0)
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'), device=self.env.device).repeat(self.cfg.num_envs)
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), True
		valid_inds = torch.zeros(self.cfg.num_envs, dtype=torch.int32, device=self.env.device)
		self._tds = []

		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Print metrics periodically
			if self._step % self.cfg.log_freq == 0:
				print_next = True

			# Reset environment
			if done.any():
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					done = torch.tensor([True] * self.cfg.num_envs)

				if self._step > 0:
					tds = torch.cat(self._tds)
					train_metrics.update(
						episode_reward=tds['reward'].nansum(0).mean(),
						# episode_success=info['success'].nanmean(),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train', print=print_next)
					print_next = False
					done_inds = done.nonzero(as_tuple=True)[0]
					self._ep_idx = self.buffer.add(tds[:-1, done_inds], valid_inds=valid_inds[done_inds])
					valid_inds[done.nonzero(as_tuple=True)] = len(self._tds) - 1

				if eval_next:
					obs = self.env.reset()
					self._tds = [self.to_td(obs)]
					valid_inds = torch.zeros(self.cfg.num_envs, dtype=torch.int32, device=self.env.device)
					eval_next = False
				else:
					# No need to reset if we didn't evaluate; bin packing env resets itself when done								
					min_valid_ind = valid_inds.min().item()
					self._tds = self._tds[min_valid_ind:]
					valid_inds -= min_valid_ind

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=done)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs
	
		self.logger.finish(self.agent)
