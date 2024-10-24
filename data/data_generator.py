# import gym_minigrid
# from gym_minigrid.wrappers import  FullyObsWrapper
# from gym_minigrid.wrappers import ImgObsWrapper

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper
from minigrid.wrappers import FullyObsWrapper
import yaml

import gymnasium as gym
# from gymnasium import Env
# import gym
import random
import os
from PIL import Image
import pdb
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_augmentor import DataAugmentor
from data.utils.controlled_reset import CustomEnvReset


'''
TODO: implement video recording stored in the src folder (only during testing)


[DONE] TODO: change the observation space based on yaml file, or output all obs together 
[DONE] TODO: variation between deterministic or stochastic actions
[DONE] TODO: implement (multiple) visual transformation on the observation output
[DONE] TODO: implement (multiple) controlled environment factors at once 
    - Random bug sometimes does not allow for factored expert state to be retrieved 
'''
class StochasticActionWrapper(gym.ActionWrapper):
    """
    Add stochasticity to the actions

    If a random action is provided, it is returned with probability `1 - prob`.
    Else, a random action is sampled from the action space.
    """

    def __init__(self, env=None, prob=0.9, random_action=None):
        super().__init__(env)
        self.prob = prob

    def action(self, action):
        """ """
        
        random = np.random.uniform()
        if random >= self.prob:
            return action
        else:
            return self.env.action_space.sample()

class DataGenerator(gym.Env):

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def __init__(self, config_filename='config.yaml'):

        super(DataGenerator, self).__init__()

        # Parse yaml file parameters for data generator
        configs = self.load_config(os.path.join(os.path.dirname(__file__), f'../configs/data_generator/{config_filename}'))
        
        # Configs from the yaml file
        self.observation_type = configs['observation_space']
        self.state_attributes = configs['state_attributes']
        self.reset_type = configs['reset_type']

        # Create the environment
        self.env = gym.make(configs['environment_name'], render_mode='rgb_array')

        # Optionally wrap the environment for fully observable states
        self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)

        self.env.max_steps = configs['max_steps']

        # Wrap the environment to enable stochastic actions
        if configs['deterministic_action'] is False:
            self.env = StochasticActionWrapper(env=self.env, prob=configs['action_stochasticity'])

        #Store the controlled factors array
        self.controlled_factors = configs['controlled_factors']

        #storing the custom reset function if needed
        self.custom_resetter = CustomEnvReset(configs['environment_name'])

        # Reset the environment to initialize it
        self.env.reset()
        
        #prepare the data augmentor instance 
        self.data_augmentor = DataAugmentor(configs['transformations'])

        #creating the observation space and actions space
        self.action_space = self.env.action_space
        
        self.observation_space = self._create_observation_space(configs['state_attribute_types'])


        #creating other gym environment attributes
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.np_random = self.env.np_random


    def _create_observation_space(self, state_attribute_types):
        #return the desired gym Spaces based on the observation space

        if self.observation_type == 'image':
            frame = self.env.get_frame(tile_size=8)
            return gym.spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8)

        elif self.observation_type == 'expert':
            
            gym_space_params = {'boolean': (0, 1, int), 'coordinate_width': (0, self.env.grid.width, int), 'coordinate_height': (0, self.env.grid.height, int), 'agent_dir': (0, 3, int)}

            relevant_state_variables = list(self._construct_state().keys())

            min_values = np.array([]); max_values = np.array([])

            for var in relevant_state_variables:
                types = state_attribute_types[var]

                for t in types:
                    
                    space_param = gym_space_params[t]

                    min_values = np.append(min_values, space_param[0])
                    max_values = np.append(max_values, space_param[1])
            
            
            return gym.spaces.Box(low=min_values, high=max_values, dtype=int)




        elif self.observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')

    def render(self):
        '''Implementing render() according to ABC of gymnasium env'''
        frame = self.env.get_frame(tile_size=8)
        frame = self.data_augmentor.apply_transformation(frame)
        return frame

    def step(self, action):
        '''
        Inputs:
        - action: integer value representing discrete actions
        0 Turn left
        1 Turn right
        2 Move forward
        3 Pick up an object
        4 Drop
        5 Toggle/activate an object
        6 Done
        '''
        (_, reward, terminated, truncated, info) = self.env.step(action)
        

        frame = self.env.get_frame(tile_size=8)

        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame
        
        #apply image transformations if needed
        frame = self.data_augmentor.apply_transformation(frame)

        state = self._construct_state()
        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the state in dictionary form for info
        info['state_dict'] = state
        #add the visual observation into the info for debugging
        info['obs'] = frame

        state = [item for sublist in state.values() for item in (sublist if isinstance(sublist, tuple) else [sublist])] 


        observation = self._get_obs(image = frame, state = state, factored = factored)
        

        # #reset in case the environment is done
        # if done:
        #     self.reset()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        '''
        Inputs: None

        Outputs:
        - observation: visual observations after resetting the environment
        - info: additional information from environment after resetting
        '''
        
        __, info = self.env.reset(seed=seed)

        #in case reset requires new state to have controlled state factors -- implement those controls
        if self.reset_type == 'custom':
            self.env = self.custom_resetter.factored_reset(self.env, self.env.unwrapped.grid.height, self.env.unwrapped.grid.width, self.controlled_factors)
        elif self.reset_type == 'random':
            self._randomize_reset()
        

        frame = self.env.get_frame(tile_size=8)
        #add the visual observation before augmentation for debugging
        info['original_obs'] = frame

        #apply image transformations if needed
        frame = self.data_augmentor.apply_transformation(frame)

        state = self._construct_state()
        factored = None

        if self.observation_type == 'factored':
            factored = self._factorize_obs(frame)

        #add the visual observation into the info for debugging
        info['obs'] = frame
        #add the state in dictionary form for info
        info['state_dict'] = state

        state = [item for sublist in state.values() for item in (sublist if isinstance(sublist, tuple) else [sublist])] 

        
        observation = self._get_obs(image = frame, state = state, factored = factored)

        
        
        return observation, info
    



        
            
        

    def _factorize_obs(self, observation):
        #TODO: implement inference time call to factored representation model
        return None
    
    def _construct_state(self):

        state = {}

        #extract the types of all tiles in the grid: useful for goal, key and door position
        types = np.array([x.type if x is not None else None for x in self.env.unwrapped.grid.grid])

        for attr in self.state_attributes:

            if hasattr(self.env.unwrapped, attr):
                state[attr] = getattr(self.env.unwrapped,attr)
            elif attr == 'goal_pos':
                state[attr] = self.env.unwrapped.grid.grid[np.where(types=='goal')[0][0]].cur_pos
            
            #other positional attributes for key and door
            elif ('key' in types) and (attr == 'key_pos'):
                state[attr] = self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]].cur_pos
            elif ('door' in types) and (attr == 'door_pos'):
                state[attr] = self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].cur_pos
            

            #other attributes like opening, holding, locked etc...
            elif ('key' in types) and (attr == 'holding_key'):
                state[attr] = int(not self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]].can_pickup())
            elif ('door' in types) and (attr == 'door_locked'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_locked)
            elif ('door' in types) and (attr == 'door_open'):
                state[attr] = int(self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]].is_open)

        
        return state
        

    def _get_obs(self, image=None, state=None, factored=None):
        
        if self.observation_type == 'image':
            return image
        
        elif self.observation_type == 'expert':
            return state
        
        elif self.observation_type == 'factored':
            return factored
        
        else:
            raise Exception('ERROR: observation type {} undefined'.format(self.observation_type))


if __name__ == '__main__':

    pdb.set_trace()

    data_generator = DataGenerator()
    data_gen_test = DataGenerator(config_filename='config_test.yaml')
    
    obs, info = data_generator.reset()
    obs_t, info_t = data_gen_test.reset()
    

    MAX_STEPS = 5

    temp_dir = os.path.relpath('./temp_obs')
    os.makedirs(temp_dir, exist_ok=True)

    for j in range(3):
        # pdb.set_trace()
        obs, info = data_generator.reset(seed=j)
        img = Image.fromarray(info['obs'])
        img.save(os.path.join(temp_dir, 'reset_test.jpeg'))

        for i in range(MAX_STEPS):

            rand_action = 6

            while (rand_action == 6):
            
                rand_action = data_generator.env.action_space.sample()

            observation, reward, terminated, truncated, info = data_generator.step(rand_action)

            print('Current State :', observation)
            print('Info: ', info['state_dict'])
            print('Reward: ', reward)

            img = Image.fromarray(info['obs'])

            img.save(os.path.join(temp_dir, '{}_modified.jpeg'.format(i)))

            # img = Image.fromarray(info['original_obs'])

            # img.save(os.path.join(temp_dir, '{}_original.jpeg'.format(i)))
        
        

