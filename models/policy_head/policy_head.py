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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
import imageio
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
import re
import pandas as pd
import wandb

import argparse

csv_logger = pd.DataFrame(columns=['train/test', 'step', 'seed', 'cumul_reward'])

from eval_callback import CustomEvalCallback

class RewardValueCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, csv_log_dir: str, num_envs:int, verbose=0, train=True):
        super(RewardValueCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.max_steps = env.env.max_steps
        self.env = env
        self.save_freq = save_freq
        self.train = train

        self.num_envs = num_envs

        self.csv_log_dir = os.path.join(csv_log_dir, 'episodic_reward_logs.csv')


        os.makedirs(csv_log_dir, exist_ok=True)

        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:

        # Save a GIF every save_freq steps
        if self.n_calls % (self.save_freq//self.num_envs) == 0:
            self._log_rewards_and_value()

            global csv_logger
            csv_logger.to_csv(self.csv_log_dir, index=False)
            

        return True
    
    
    def _log_rewards_and_value(self):
        # Reset the environment
        obs, info = self.env.reset()
        dones = False
        step = 0
        cumulative_reward = 0

        while not dones.all() and step < self.max_steps:
            # Predict action and get value function (self.model.predict returns the action and value)
            actions, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, __, infos = self.env.step(actions)

        while not dones and step < self.max_steps:
            # Predict action and get value function (self.model.predict returns the action and value)
            actions, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, __, infos = self.env.step(actions)

            # Log the value function and reward to TensorBoard
            cumulative_reward += rewards
            self.writer.add_scalar(f"{'train' if self.train else 'test'} reward per step", rewards, self.num_timesteps + step)
            step += 1

        
        #NOTE: error check -- ensuring cumulative reward isn't > 1 or < 0
        assert (cumulative_reward <= 1.0 and cumulative_reward >= 0.0), f'ERROR: cumulative reward is {cumulative_reward}'

        # Log the cumulative reward
        self.writer.add_scalar(f"{'train' if self.train else 'test'} cumul reward", cumulative_reward, self.num_timesteps)
        # self.writer.add_scalar(f"{'train' if self.train else 'test'} avg reward", cumulative_reward/step, self.num_timesteps)

        # Print the logged rewards
        print("=-------------------------=")
        print(f"{'train' if self.train else 'test'} cumul reward @ {self.num_timesteps}: ", cumulative_reward)
        # print(f"{'train' if self.train else 'test'} avg reward @ {self.num_timesteps}: ", cumulative_reward/step)

        #log the cumulative rewards to csv file
        global csv_logger
        csv_logger = pd.concat([pd.DataFrame([{'train/test': 'train' if self.train else 'test', 'step': self.num_timesteps, 'seed': self.model.seed, 'cumul_reward': cumulative_reward}]), csv_logger], ignore_index=True)
        
    
    def _on_training_end(self):
        self.writer.close()

class GifLoggingCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, num_envs: int, name_prefix: str = "", verbose=1):
        super(GifLoggingCallback, self).__init__(verbose)
        self.env = env
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.name_prefix = name_prefix

        

        self.num_envs = num_envs
        os.makedirs(self.log_dir, exist_ok=True)
        # self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        
        # Save a GIF every save_freq steps
        if self.n_calls % (self.save_freq//self.num_envs) == 0:
            self._create_gif()
            self._create_value_func()
            
        return True
    
    def _create_value_func(self):
        '''extract the value function for a particular observation'''
        
        original_obs, info = self.env.reset()

        value_function = np.zeros((self.env.env.unwrapped.width, self.env.env.unwrapped.height)) * np.nan


        for w in range(self.env.env.unwrapped.width):
            for h in range(self.env.env.unwrapped.height):

                obj = self.env.env.unwrapped.grid.get(w, h)

                if obj is None:

                    # place the agent here
                    self.env.env.unwrapped.agent_pos = (w, h)

                    value_estim = []
                    
                    for dir in range(4):
                        
                        self.env.env.unwrapped.agent_dir = dir
                        obs = self.env.env.unwrapped.get_frame(tile_size=8)

                       
                        
                        obs_tensor, vector_env = self.model.policy.obs_to_tensor(obs)

                        if callable(getattr(self.model.policy, "predict_values", None)):
                            value = self.model.policy.predict_values(obs_tensor).item()
                        elif callable(getattr(self.model, "q_net", None)):
                            value = torch.sum(self.model.q_net(obs_tensor), dim=1).item()

                        value_estim.append(value)

                    value_function[w, h] = np.mean(value_estim)
        
        
        #normalize the value function so it is 0-1
        value_function = value_function / np.nansum(value_function)

        #plotting value function over the observation
        fig = plt.figure(frameon=False)
        plt.imshow(original_obs)
        plot_extent = [0, self.env.env.unwrapped.width * 8, self.env.env.unwrapped.height * 8, 0]
        plt.imshow(value_function, alpha=0.5, cmap='viridis', extent = plot_extent)
        plt.colorbar()
        plt.title(f"Value Function @ Step {self.num_timesteps}")
        plt.savefig(os.path.join(self.log_dir, f"{self.name_prefix}_value_step_{self.num_timesteps}.png"))
        plt.close()


    def _create_gif(self):
        images = []
        obs = self.env.reset()
        dones = np.zeros(self.env.num_envs, dtype=bool)
        step = 0
        
        while not done:
            # Render the environment and save frame
            frame = self.env.render()
            images.append(frame)
            
            # Take a step in the environment
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            step += 1

        #save the final frame of +ve reward
        frame = self.env.render()
        images.append(frame)
       

        gif_path = os.path.join(self.log_dir, f"{self.name_prefix}_policy_step_{self.num_timesteps}.gif")
        imageio.mimsave(gif_path, [np.array(img) for img in images], fps=5)

        
        # vid_tensor: (N,T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        #self.writer.add_video(f"{self.name_prefix}_policy_step_{self.num_timesteps}", vid_tensor, global_step=self.num_timesteps, fps=4, walltime=None)

    def _log_gif_to_tensorboard(self, gif_path: str):
        # Read the GIF as bytes and log to TensorBoard
        with open(gif_path, 'rb') as f:
            gif_bytes = f.read()

        self.writer.add_image(f"{self.name_prefix}_policy_step_{self.num_timesteps}", gif_bytes, self.num_timesteps, dataformats="HWC")

    def _on_training_end(self):
        plt.close()

def gen_env(seed = None, config='config.yaml'):
    env = Monitor(DataGenerator(config))
    env.reset(seed=seed)
    return env


class PolicyHead:
    def __init__(self, model_config_path, data_config_path, seed=None):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()

        #set the seed in order to create argparsable separate runs for each seed
        self.seed = self.model_config['seed'] if seed is None else seed

        print('POLICY NAME: ', self.policy_name)
        self.parallel_train_env = VecVideoRecorder(
            self.create_parallel_envs(seed = self.seed), 
            f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{self.seed}/", 
            record_video_trigger=lambda x: x % self.model_config['save_weight_freq'] == 0, 
            video_length=1000, 
            name_prefix='policy_video'
        )
        self.valid_env = self.create_parallel_envs(seed = self.seed)
        self.eval_env = self.create_parallel_envs(seed = self.seed, train=False)
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
    
    def create_parallel_envs(self, seed: int=0, train=True, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.model_config['num_parallel_envs']
        if train:
            return SubprocVecEnv([lambda: gen_env(seed + i, 'config.yaml') for i in range(num_parallel)])
        else:
            return SubprocVecEnv([lambda: gen_env(seed + i, 'config_test.yaml') for i in range(num_parallel)])
        

    def create_models(self, seed: int = 0):
        if self.algorithm == "PPO":
            ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}

            #NOTE: include lr schedule if needed
            #lr_schedule = self.linear_schedule(self.model_config['learning_rate'])                
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
            entity='waymao', 
            name=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}_seed_{self.seed}',
            group=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}',
            sync_tensorboard=True,
            monitor_gym=True
        )
        train_interval = self.model_config['train_interval']

        if os.path.exists(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}") and len(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}")) > 0:
            latest_weight = list(sorted(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}")))
            
            latest_weight = max(latest_weight, key=lambda f: int(re.search(r'\d+', f.split('step_')[1]).group()))

            final_path = os.path.join(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}", latest_weight.split('.')[0])

            self.model.load(path = final_path, env = self.parallel_train_env)

            
        
        # Use Built-in Eval Callback to support multiple parallel environments
        reward_validation_callback = CustomEvalCallback("validation", eval_env=self.valid_env, n_eval_episodes=10, eval_freq=self.model_config['reward_log_freq'])
        reward_eval_callback = CustomEvalCallback("eval", eval_env=self.eval_env, n_eval_episodes=10, eval_freq=self.model_config['reward_log_freq'])
        # reward_callback = RewardValueCallback(env = self.train_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{self.seed}/", csv_log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_rewards/{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], train=True)
        # eval_reward_callback = RewardValueCallback(env = self.eval_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{self.seed}/", csv_log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_rewards/{self.data_config['observation_space']}/seed_{self.seed}/", num_envs = self.model_config['num_parallel_envs'], train=False)
        
        # VecVideoRecorder is used instead of GifLoggingCallback
        # gif_callback = GifLoggingCallback(env = self.valid_env, save_freq = self.model_config['gif_log_freq'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], name_prefix = 'policy_gif')
        checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{self.seed}/", name_prefix=f'{self.algorithm}_seed{self.seed}_step', save_replay_buffer=True)


        # Create the callback list
        callback = CallbackList([reward_validation_callback, reward_eval_callback, checkpoint_callback])
        
        
        self.model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{self.seed}', progress_bar = True, reset_num_timesteps=False, callback = callback)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    
    
    print(DataGenerator('config.yaml').observation_space)
    print(DataGenerator('config_test.yaml').observation_space)
    # policy_head = PolicyHead(
    #     SubprocVecEnv([lambda: gen_env(i) for i in range(4)]), 
    #     SubprocVecEnv([lambda: gen_env(i) for i in range(5)]), 
    #     SubprocVecEnv([lambda: gen_env(i, config_path='config_test.yaml') for i in range(5)]), 
    policy_head = PolicyHead( 
        'configs/models/config.yaml', 
        'configs/data_generator/config.yaml',
        seed=args.seed
    )
    policy_head.train_and_evaluate_policy()

