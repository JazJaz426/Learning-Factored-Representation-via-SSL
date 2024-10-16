import gym_minigrid
from gym_minigrid.wrappers import  FullyObsWrapper
from gym_minigrid.wrappers import ImgObsWrapper
import yaml

# import gymnasium
import gym
import random
import os
from PIL import Image
import pdb
import numpy as np
from data_augmentor import DataAugmentor

'''
TODO: implement video recording stored in the src folder (only during testing)
[IMP] TODO: implement (multiple) controlled environment factors at once 
TODO: implement randomized resetting of attribute features

[DONE] TODO: change the observation space based on yaml file, or output all obs together 
[DONE] TODO: variation between deterministic or stochastic actions
[DONE] TODO: implement (multiple) visual transformation on the observation output
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

    def __init__(self):

        # Parse yaml file parameters for data generator
        configs = self.load_config('../configs/data_generator/config.yaml')
        
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
            self.env = StochasticActionWrapper(env= self.env, prob = configs['action_stochasticity'])

        #Store the controlled factors array
        self.controlled_factors = configs['controlled_factors']

        # Reset the environment to initialize it
        self.env.reset()
        
        #prepare the data augmentor instance 
        self.data_augmentor = DataAugmentor(configs['transformations'])



        
        



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

        (_, reward, terminated, _, info) = self.env.step(action)
        

        frame = self.env.render()

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

        return observation, reward, terminated, None, info

    def reset(self):
        '''
        Inputs: None

        Outputs:
        - observation: visual observations after resetting the environment
        - info: additional information from environment after resetting
        '''
        
        self.env.reset()

        #in case reset requires new state to have controlled state factors -- implement those controls
        if self.reset_type == 'custom':
            self._control_factors()
        elif self.reset_type == 'random':
            self._randomize_reset()
        

        frame = self.env.render()
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
    
    def _randomize_reset(self):
        raise NotImplementedError('ERROR: not implemented yet')

    def _control_factors(self):
        #fix each of the controlled factors using the unwrapped state variable


        #extract the types of all tiles in the grid: useful for goal, key and door position
        types = np.array([x.type if x is not None else None for x in self.env.unwrapped.grid.grid])

        for factor in self.controlled_factors.keys():

            if factor == 'agent_pos':
                self.env.unwrapped.place_agent(top = self.controlled_factors[factor], size=(1,1))
            elif factor == 'agent_dir':
                None
            elif factor == 'goal_pos':
                self.env.unwrapped.place_obj(obj = self.env.unwrapped.grid.grid[np.where(types=='goal')[0][0]], top=self.controlled_factors[factor], size=(1,1))
            
            elif ('key' in types) and (factor == 'key_pos'):
                self.env.unwrapped.place_obj(obj = self.env.unwrapped.grid.grid[np.where(types=='key')[0][0]], top=self.controlled_factors[factor], size=(1,1))
            elif ('door' in types) and (factor == 'door_pos'):
                self.env.unwrapped.place_obj(obj = self.env.unwrapped.grid.grid[np.where(types=='door')[0][0]], top=self.controlled_factors[factor], size=(1,1))

            elif ('key' in types) and (factor == 'holding_key'):
                None
            elif ('door' in types) and (factor == 'door_locked'):
                None
            elif ('door' in types) and (factor == 'door_open'):
                None
            
        

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

    MAX_STEPS = 5

    temp_dir = os.path.relpath('./temp_obs')
    os.makedirs(temp_dir, exist_ok=True)

    for j in range(3):
        for i in range(MAX_STEPS):

            rand_action = 6

            while (rand_action == 6):
            
                rand_action = data_generator.env.action_space.sample()

            observation, reward, terminated, __, info = data_generator.step(rand_action)

            print('Current State :', observation)
            print('Info: ', info['state_dict'])

            img = Image.fromarray(info['obs'])

            img.save(os.path.join(temp_dir, '{}_modified.jpeg'.format(i)))

            img = Image.fromarray(info['original_obs'])

            img.save(os.path.join(temp_dir, '{}_original.jpeg'.format(i)))
        
        pdb.set_trace()
        data_generator.reset()

