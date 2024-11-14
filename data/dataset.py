import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))
import logging
import torch
from torch.utils.data import Dataset
try:
    from data_generator import DataGenerator
except:
    logging.warning("Using relative import for data_generator")
    from .data_generator import DataGenerator
import numpy as np

import pdb
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_env_config, limit, policy_model = None, model_path = None, mode = 'seq'):
        
        self.data_env = DataGenerator(data_env_config)
        self.count = 0
        self.limit = limit
        self.mode = mode #options: seq [sequential], cont [controlled factors], triplet [triplet pair with different actions], rand [random reset, inbuilt]
        self.classes = list(range(1000))

        #policy model used for data generation: load from checkpoint path if needed
        if policy_model is not None:
            self.model = policy_model
            self.model.load(path = model_path, env = self.data_env)
        else:
            self.model = RandomPolicy(self.data_env)

    def __len__(self):
        return self.limit

    def sample_factors(self):

        #uniformly sample factors for the environment
        sampled_factors = {}
        valid = False

        while not valid:
            
            #do sampling process between min max uniformly
            for attr in self.data_env.state_attributes:

                low, high, typ = self.data_env.get_low_high_attr(attr)
                rand_val = typ(np.random.uniform(low, high))

                sampled_factors[attr] = rand_val


            valid, err = self.data_env.custom_resetter.check_valid_factors(self.data_env.env, sampled_factors)

        return sampled_factors



    def __getitem__(self, index):

        if index >= self.limit:
            raise IndexError("Index out of range")

        if self.mode == 'seq':

            #get the current visual observation and underlying state
            obs = self.data_env.get_curr_obs()
            state, norm_state = self._construct_state()
            
            #predict action and take a step in the environment
            actions, __ = self.model.predict(obs, deterministic=True)
            self.data_env.step(actions)

        elif self.mode == 'cont':
            #NOTE: input controlled_factors as empty dictionary so that all factors are randomized following env rules
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, {})
            logging.info("after reset")
            #get the current visual observation and underlying state
            obs = self.data_env.get_curr_obs()
            state, norm_state = self.data_env._construct_state()
        elif self.mode == 'rand':

            obs, info = self.data_env.reset()
            state, norm_state = self.data_env._construct_state()

        elif self.mode == 'triplet':
            raise NotImplementedError(f'ERROR: data generation mode cannot be {self.mode}')
        else:
            raise NotImplementedError(f'ERROR: data generation mode cannot be {self.mode}')

        self.count += 1

        return obs, norm_state, state 


class RandomPolicy:

    def __init__(self, env):

        self.action_space = env.action_space
    
    def predict(self, obs, deterministic=True):

        return self.action_space.sample(), None



if __name__ == "__main__":

    dataset = CustomDataset('configs/data_generator/config.yaml', limit=30, mode='cont')

    pdb.set_trace()
    from PIL import Image

    os.makedirs('./temp', exist_ok = True)

    for i,d in enumerate(dataset):

        img = Image.fromarray(d[0])
        img.save(f'./temp/{i+1}.png')
        print(i+1)
        print(dataset.count)
        print(dataset.limit)
        print('-------')
