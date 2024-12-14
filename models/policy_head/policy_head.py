import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.data_generator import DataGenerator
import pdb
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
import imageio
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
import re
import pandas as pd
import wandb

import argparse



from custom_callbacks import CustomEvalCallback, CustomVideoRecorder, RewardValueCallback, ValuePlottingCallback


class PolicyHead:
    def __init__(self, model_config, data_config, data_test_config, seed=None):
        self.model_config = model_config
        self.data_config = data_config
        self.data_test_config = data_test_config
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()

        #set the seed in order to create argparsable separate runs for each seed
        self.seed = self.model_config['seed'] if seed is None else seed

        print('POLICY NAME: ', self.policy_name)
        

        self.parallel_train_env = VecVideoRecorder(
            self.create_parallel_envs(seed = self.seed),
            f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{self.seed}/", 
            record_video_trigger=lambda x: x % (self.model_config['video_log_freq'] // self.model_config['num_parallel_envs']) == 0, 
            video_length=self.model_config['video_length'], 
            name_prefix=self.policy_name
        )

        # self.parallel_train_env = self.create_parallel_envs(seed = self.seed)

        self.valid_env = self.create_parallel_envs(seed = self.seed)
        self.eval_env = self.create_parallel_envs(seed = self.seed, train=False)

        
        self.dummy_env = self.create_env(seed=self.seed, config=data_config)()


        self.model = self.create_models(seed=self.seed)

        #check that critical configs for test and train are equal 
        assert (self.valid_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.valid_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

       
        assert (self.parallel_train_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.parallel_train_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

    def linear_schedule(self, initial_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    def select_policy(self):
        if self.data_type == "image":
            return "CnnPolicy"
        elif self.data_type in ["factored", "expert"]:
            return "MlpPolicy"
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def create_env(self, seed = None, config=None):
        def _init():
            env = Monitor(DataGenerator(config))
            env.reset(seed=seed)
            return env
        return _init

    
    def create_parallel_envs(self, seed: int=0, train=True, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.model_config['num_parallel_envs']
        if train:
            return SubprocVecEnv([self.create_env(seed, self.data_config) for _ in range(num_parallel)])
        else:
            return SubprocVecEnv([self.create_env(seed, self.data_test_config) for _ in range(num_parallel)])
        

    def create_models(self, seed: int = 0):
        if self.algorithm == "PPO":
            ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}

            #NOTE: include lr schedule if needed
            #lr_schedule = self.linear_schedule(self.model_config['learning_rate'])
            if self.data_type == "factored":
                ppo_params['policy_kwargs'] = dict(
                    net_arch=[dict(pi=[64], vf=[64])]
                )
            model = PPO(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                **ppo_params,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/"
            )
        elif self.algorithm == "DQN":
            dqn_params = {k: v for k, v in self.model_config['dqn'].items() if v is not None}
            model = DQN(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                **dqn_params,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/"
            )
        elif self.algorithm == "A2C":
            a2c_params = {k: v for k, v in self.model_config['a2c'].items() if v is not None}
            model = A2C(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                **a2c_params,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/"
            )
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        return model
    

    def train_and_evaluate_policy(self):
        wandb.init(
            project='ssl_rl',
            entity='ssl-factored-reps', 
            name=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}_seed_{self.seed}',
            group=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}',
            sync_tensorboard=True,
            monitor_gym=True,
            config={
                "model": self.model_config,
                "data": self.data_config,
                "seed": self.seed
            }
        )
        train_interval = self.model_config['train_interval']

        if os.path.exists(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}") and len(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}")) > 0:
            try:
                #fetch the best weights for the model rather than latest
                best_weight = os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/best_weight")[0].split('.')[0]
                final_path = os.path.join(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/best_weight", best_weight)

                self.model.load(path = final_path, env = self.parallel_train_env)
            except:
                pass

            
        # Use Built-in Eval Callback to support multiple parallel environments
        reward_validation_callback = CustomEvalCallback("validation", eval_env=self.valid_env, n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/best_weight")
        reward_eval_callback = CustomEvalCallback("eval", eval_env=self.eval_env, n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = None)
       
        # VecVideoRecorder is used instead of GifLoggingCallback
        value_callback = ValuePlottingCallback(env = self.dummy_env, save_freq = self.model_config['video_log_freq'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], name_prefix = f'{self.policy_name}_policy_value')
        checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq']//self.model_config['num_parallel_envs'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/", name_prefix=f'{self.algorithm}_seed{self.seed}_step', save_replay_buffer=True)


        # Create the callback list
        callback = CallbackList([reward_validation_callback, reward_eval_callback, value_callback, checkpoint_callback])
        
        self.model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{self.seed}', progress_bar = True, reset_num_timesteps=False, callback = callback)

        if self.model_config['wandb_log']:
            wandb.finish()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


import collections.abc

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_params(params: list):
    # parse nested parameters, such as --field1.field1b value
    parsed_params = {}
    for i in range(0, len(params), 2):
        curr_params = parsed_params
        keys = params[i].lstrip('--').split('.')
        for key in keys[:-1]:
            if key not in curr_params:
                curr_params[key] = {}
            curr_params = curr_params[key]
        curr_params[keys[-1]] = params[i+1]
    return parsed_params
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('params', nargs='*', help='Parameters to update in the config file in the form of --field1.field1b value')
    args = parser.parse_args()
    
    params = parse_params(args.params)
    print("parsed additional params:", params)
    
    model_config = load_config('configs/models/config.yaml')['policy_head']
    data_config = update(load_config('configs/data_generator/config.yaml'), params)
    data_test_config = update(load_config('configs/data_generator/config_test.yaml'), params)
    
    assert model_config is not None
    assert data_config is not None
    assert data_test_config is not None
  
    policy_head = PolicyHead( 
        model_config, data_config, data_test_config,
        seed=args.seed
    )
    policy_head.train_and_evaluate_policy()

