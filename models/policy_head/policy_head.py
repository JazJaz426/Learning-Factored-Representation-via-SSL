import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor
from models.utils.impala_cnn import ImpalaCNN
from models.utils.flatten_mlp import FlattenMLP
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
import numpy as np

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
import pdb


from custom_callbacks import CustomEvalCallback, CustomVideoRecorder, RewardValueCallback, ValuePlottingCallback, SupervisedEncoderCallback


class PolicyHead:
    def __init__(self, model_config_path, data_config_path, seed=None):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        
        self.additional_params = dict(
            stop_gradient = True if self.model_config['representation_learner'] != 'direct' else False
        )
        
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

        
        self.dummy_env = self.create_env(seed=self.seed)()


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

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def select_policy(self):
        if self.data_type == "image":
            return "CnnPolicy"
        elif self.data_type in ["factored", "expert"]:
            return "MlpPolicy"
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def create_env(self, seed = None, config='config.yaml'):
        

        def _init():
            env = Monitor(DataGenerator(config))
            env.reset(seed=seed)
            return env

        return _init

    
    def create_parallel_envs(self, seed: int=0, train=True, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.model_config['num_parallel_envs']
        if train:
            vecenv =  SubprocVecEnv([self.create_env(seed, 'config.yaml') for _ in range(num_parallel)])
        else:
            vecenv = SubprocVecEnv([self.create_env(seed, 'config_test.yaml') for _ in range(num_parallel)])
        
        #add self transposition to (C, H, W) if image observation space
        if len(vecenv.observation_space.shape) > 1:
            vecenv = VecTransposeImage(vecenv)
        
        return vecenv

    def create_models(self, seed: int = 0):
        if self.algorithm == "PPO":
            ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}

            #TODO: maybe make this more elegant?
            expert_obs = self.dummy_env.expert_observation_space

            
            policy_kwargs = dict(
                net_arch = dict(pi=self.model_config['ppo_policy_kwargs']['pi_dims'], vf=self.model_config['ppo_policy_kwargs']['vf_dims']),
                features_extractor_class = ImpalaCNN if len(self.parallel_train_env.observation_space.shape) > 1 else FlattenMLP,
                features_extractor_kwargs = dict(features_dim = self.model_config['ppo_policy_kwargs']['features_dim'], vector_size_per_factor = self.model_config['vector_size_per_factor'], expert_obs = expert_obs, stop_gradient = self.additional_params['stop_gradient'])
            )
            
            #NOTE: include lr schedule if needed
            #lr_schedule = self.linear_schedule(self.model_config['learning_rate'])   
            model = PPO(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/",
                # policy_kwargs = policy_kwargs,
                **ppo_params,
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
            project='disentangled_rep',
            entity='ssl-factored-reps', 
            name=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}_seed_{self.seed}',
            group=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}',
            sync_tensorboard=True,
            monitor_gym=True,
            config={
                "model": self.model_config,
                "data": self.data_config,
                "seed": self.seed,
                "num_parallel_envs": self.model_config['num_parallel_envs']
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
        value_callback = ValuePlottingCallback(env = self.dummy_env, save_freq = self.model_config['video_log_freq']//self.model_config['num_parallel_envs'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], name_prefix = f'{self.policy_name}_policy_value')
        checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq']//self.model_config['num_parallel_envs'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/", name_prefix=f'{self.algorithm}_seed{self.seed}_step', save_replay_buffer=True)

        
        

        # Create the callback list
        callbacks = CallbackList([reward_validation_callback, reward_eval_callback, value_callback, checkpoint_callback])
        
        #If using supervised learning or our approach, create separet SupervisedEncoderCallback for supervised learning approach
        if self.model_config['representation_learner'] == 'supervised':
            supervised_encoder_callback = SupervisedEncoderCallback(custom_name = "supervised")
            callbacks.callbacks.append(supervised_encoder_callback)

        self.model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{self.seed}', progress_bar = True, reset_num_timesteps=False, callback = callbacks)

        if self.model_config['wandb_log']:
            wandb.finish()
    
    def evaluate_policy(self, min_seed:int, max_seed:int, transform_obs:bool, transform_func):
        
        #collect all rewards and dones
        tot_rewards = []
        tot_steps = []
        tot_dones = []
        
        #iterate through all seeds and evaluate policy
        assert max_seed >= min_seed, 'ERROR: the max_seed needs to be > min_seed'

        for seed in range(min_seed, max_seed+1):
            self.valid_env = self.create_parallel_envs(seed = seed)
            obs = self.valid_env.reset()
            for i in range(self.data_config['max_steps']):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = vec_env.step(action)

                tot_rewards.append(sum(rewards))
                tot_dones.append(sum(dones))

                if dones:
                    break
            
            tot_steps.append(i)
            
        return np.mean(tot_rewards), np.std(tot_rewards), np.mean(tot_dones), np.std(tot_dones), np.mean(tot_steps), np.std(tot_steps)
        

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    
    
    print(DataGenerator('config.yaml').observation_space)
    print(DataGenerator('config_test.yaml').observation_space)
  
    policy_head = PolicyHead( 
        'configs/models/config.yaml', 
        'configs/data_generator/config.yaml',
        seed=args.seed
    )
    policy_head.train_and_evaluate_policy()
    


