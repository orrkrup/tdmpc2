from isaac_robot_sims.envs.bin_packing import OnlineBinPackingEnv
from isaac_robot_sims.definitions import ItemDescription, BinDescription, ObservationDescription

import gymnasium as gym
import torch


class PointCloudWrapper(gym.Wrapper):
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
        obs = self.env.reset(**kwargs)
        return self._get_obs(obs)
    
    def step(self, action):
        if 1 == self.num_envs:
            action = action.unsqueeze(0)
    
        obs, reward, done, info = self.env.step(action)
        if 'success' not in info.keys():
            info['success'] = 0

        if 1 == self.num_envs:
            reward = reward.item()    
        return self._get_obs(obs), reward, done, info
	
    def pad_pc(self, pc, n_points=2048):
        self.num_total += 1
        if pc.shape[0] >= n_points:
            return pc[:n_points]
        else:
            self.num_padded += 1
            return torch.cat([pc, torch.zeros(n_points - pc.shape[0], pc.shape[1], device=pc.device)], dim=0)
	

def make_env(cfg):

    # TODO: get desc details from cfg
    
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    # env setup
    # item_desc = ItemDescription(type='irbpp', dataset_name='blockout', dataset_root='../isaac_robot_sims/data/IR_BPP_Dataset/')
    # bin_desc = BinDescription(size=[0.32, 0.32, 0.3])

    item_desc = ItemDescription(type='cuboid', n_items=60, items_to_pack=30, test_items=30, 
                                mode='random', limits={'x': [0.1, 0.1], 'y': [0.1, 0.1], 'z': [0.1, 0.1]})
    bin_desc = BinDescription(size=[0.5, 0.3, 0.2])
    obs_desc = ObservationDescription(n_video_envs=1)
    env = OnlineBinPackingEnv(num_envs=cfg.num_envs, headless=True, item_desc=item_desc, bin_desc=bin_desc, obs_desc=obs_desc, device=device_name)

    env = PointCloudWrapper(cfg, env, use_object=False)
    return env