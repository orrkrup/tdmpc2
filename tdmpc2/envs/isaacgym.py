from isaac_robot_sims.envs.bin_packing import PlanningBinPackingEnv
from isaac_robot_sims.definitions import ItemDescription, BinDescription, ObservationDescription

import gymnasium as gym
import torch


class BinPackingWrapper(gym.Wrapper):
    """
	Wrapper for pixel observations. Compatible with DMControl environments.
	"""
    def __init__(self, cfg, env, use_object=False):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.use_object = use_object
        self.num_envs = env.num_envs
		
        if self.use_object:
            self.observation_space = {f'{k}_pc': v for k, v in self.env.observation_space.items()}
        else:
            self.observation_space = self.env.observation_space['bin']

    def _get_obs(self, obs):
        if not self.use_object:
            obs = obs['bin']
            if len(obs) == 1:
                obs = obs[0]
        return obs
    
    def reset(self, **kwargs):
        if 'ids' in kwargs.keys() and 'options' not in kwargs.keys():
            kwargs['options'] = {'ids': kwargs['ids']}
        obs = self.env.reset(**kwargs)

        num_actions = 1
        obs = {k: v[:num_actions] for k, v in obs.items()}
        return self._get_obs(obs)
    
    def rand_act(self, use_all=False):
        if isinstance(self.unwrapped, PlanningBinPackingEnv) and not use_all:
            return self.env.rand_act()
        else:
            return torch.stack([self.env.rand_act() for _ in range(self.num_envs)], dim=0)
    
    def step(self, action, **kwargs):

        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        num_actions = action.shape[0]
        if num_actions < self.num_envs:
            action = torch.cat([action, torch.zeros(self.num_envs - num_actions, action.shape[1], device=action.device)], dim=0)

        obs, reward, done, info = self.env.step(action, **kwargs)

        if num_actions < self.num_envs:
            if obs is not None:
                obs = {k: v[:num_actions] for k, v in obs.items()}
            reward = reward[:num_actions]
            done = done[:num_actions]

        if obs is not None:
            obs = self._get_obs(obs)
        
        if 'success' not in info.keys():
            info['success'] = 0

        if 1 == num_actions:
            reward = reward.item()
            done = done.item()
        return obs, reward, done, info
	
    def render(self):
        vid_im = self.env.render()[0]
        # TODO: might need to manipulate returned image here to match video recording format
        return vid_im
    
    def pad_pc(self, pc, n_points=2048):
        # TODO: move into actual env
        self.num_total += 1
        if pc.shape[0] >= n_points:
            return pc[:n_points]
        else:
            self.num_padded += 1
            return torch.cat([pc, torch.zeros(n_points - pc.shape[0], pc.shape[1], device=pc.device)], dim=0)
	



def make_env(cfg):

    # TODO: get desc details from cfg
    device_name = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    # env setup
    # item_desc = ItemDescription(type='irbpp', dataset_name='blockout', dataset_root='../isaac_robot_sims/data/IR_BPP_Dataset/')
    # bin_desc = BinDescription(size=[0.32, 0.32, 0.3])

    item_desc = ItemDescription(type='cuboid', n_items=60, items_to_pack=30, test_items=30, 
                                mode='random', limits={'x': [0.095, 0.095], 'y': [0.095, 0.095], 'z': [0.095, 0.095]})
    bin_desc = BinDescription(size=[0.5, 0.3, 0.2])
    # item_desc = ItemDescription(type='irbpp', dataset_name='blockout', dataset_root='/home/orr/research/isaac_robot_sims/data/IR_BPP_Dataset/')
    # bin_desc = BinDescription(size=[0.32, 0.32, 0.3])
    obs_desc = ObservationDescription(n_video_envs=1)
    env = PlanningBinPackingEnv(num_envs=cfg.num_envs, headless=True, item_desc=item_desc, bin_desc=bin_desc, obs_desc=obs_desc, device=device_name)

    env = BinPackingWrapper(cfg, env, use_object=False)
    return env