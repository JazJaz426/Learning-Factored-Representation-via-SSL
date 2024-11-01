import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.data_generator import DataGenerator
import pdb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import imageio
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
import re
import pandas as pd

class RewardValueCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, csv_log_dir: str, verbose=0, train=True):
        super(RewardValueCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.max_steps = env.env.max_steps
        self.env = env
        self.save_freq = save_freq
        self.train = train

        self.csv_log_dir = os.path.join(csv_log_dir, 'episodic_reward_logs.csv')
        self.csv_logger = pd.DataFrame(columns=['train/test', 'step', 'seed', 'cumul_reward'])

        os.makedirs(csv_log_dir, exist_ok=True)

        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Save a GIF every save_freq steps
        
        if self.n_calls % self.save_freq == 0:
            self._log_rewards_and_value()
            self.csv_logger.to_csv(self.csv_log_dir, index=False)
            

        return True
    
    
    def _log_rewards_and_value(self):
        # Reset the environment
        obs, info = self.env.reset()
        done = False
        step = 0
        cumulative_reward = 0

        with torch.no_grad():

            while not done and step < self.max_steps:
                # Predict action and get value function (self.model.predict returns the action and value)
                actions, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, __, infos = self.env.step(actions)

                # Log the value function and reward to TensorBoard
                cumulative_reward += rewards
                self.writer.add_scalar(f"{'train' if self.train else 'test'} reward per step", rewards, self.num_timesteps + step)
                step += 1

                #if goal is achieved then do not continue iterations
                if dones:
                    break

            
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
            self.csv_logger[len(self.csv_logger)] = {'train/test': 'train' if self.train else 'test', 'step': self.num_timesteps, 'seed': self.model.seed, 'cumul_reward': cumulative_reward}
        
    
    def _on_training_end(self):
        self.writer.close()

class GifLoggingCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, max_steps: int = 100, name_prefix: str = "", verbose=1):
        super(GifLoggingCallback, self).__init__(verbose)
        self.env = env
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.name_prefix = name_prefix

        os.makedirs(self.log_dir, exist_ok=True)
        # self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        # Save a GIF every save_freq steps
        if self.n_calls % self.save_freq == 0:
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
        
        pdb.set_trace()
        #normalize the value function so it is 0-1
        value_function = value_function / np.sum(value_function)

        #plotting value function over the observation
        fig = plt.figure(frameon=False)
        plt.imshow(original_obs)
        plot_extent = [0, self.env.env.unwrapped.width * 8, self.env.env.unwrapped.height * 8, 0]
        plt.imshow(value_function, alpha=0.5, cmap='viridis', extent = plot_extent)
        plt.colorbar()
        plt.title(f"Value Function @ Step {self.num_timesteps}")
        plt.savefig(os.path.join(self.log_dir, f"{self.name_prefix}_value_step_{self.num_timesteps}.png"))


    def _create_gif(self):
        images = []
        obs, info = self.env.reset()
        done = False
        step = 0
        # pdb.set_trace()
        
        while not done and step < self.max_steps:
            # Render the environment and save frame
            frame = self.env.render()
            images.append(frame)
            
            # Take a step in the environment
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done,___,info = self.env.step(action)
            step += 1

        #save the final frame of +ve reward
        frame = self.env.render()
        images.append(frame)
       

        gif_path = os.path.join(self.log_dir, f"{self.name_prefix}_policy_step_{self.num_timesteps}.gif")
        # pdb.set_trace()
        imageio.mimsave(gif_path, [np.array(img) for img in images], fps=5)

        
        # vid_tensor: (N,T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        #self.writer.add_video(f"{self.name_prefix}_policy_step_{self.num_timesteps}", vid_tensor, global_step=self.num_timesteps, fps=4, walltime=None)

    def _log_gif_to_tensorboard(self, gif_path: str):
        # Read the GIF as bytes and log to TensorBoard
        with open(gif_path, 'rb') as f:
            gif_bytes = f.read()

        self.writer.add_image(f"{self.name_prefix}_policy_step_{self.num_timesteps}", gif_bytes, self.num_timesteps, dataformats="HWC")

    def _on_training_end(self):
        self.writer.close()

class PolicyHead:
    def __init__(self, 
                 train_env, 
                 test_env,
                 eval_env, 
                 model_config_path, 
                 data_config_path
        ):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()

        print('POLICY NAME: ', self.policy_name)
        self.train_env = train_env
        self.test_env = test_env
        self.eval_env = eval_env
        self.models = self.create_models(num_models=self.model_config['num_models'])

        #check that critical configs for test and train are equal 
        assert (self.train_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.train_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

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

    def create_models(self, num_models: int = 3):
        models = {}
        for model_num in range(num_models):
            if self.algorithm == "PPO":
                ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}

                #NOTE: include lr schedule if needed
                #lr_schedule = self.linear_schedule(self.model_config['learning_rate'])                
                models[model_num] = PPO(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **ppo_params,
                    tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{model_num}/"
                )
            elif self.algorithm == "DQN":
                dqn_params = {k: v for k, v in self.model_config['dqn'].items() if v is not None}
                models[model_num] = DQN(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **dqn_params,
                    tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{model_num}/"
                )
            elif self.algorithm == "A2C":
                a2c_params = {k: v for k, v in self.model_config['a2c'].items() if v is not None}
                models[model_num] = A2C(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **a2c_params,
                    tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{model_num}/"
                )
            else:
                raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        return models
    

    def train_and_evaluate_policy(self, total_timestamps):

        
        
        train_interval = self.model_config['train_interval']
        # eval_interval = self.model_config['eval_interval']
        iteration = 0
        for seed in self.models.keys():

            if os.path.exists(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{seed}") and len(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{seed}")) > 0:
                latest_weight = list(sorted(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{seed}")))
                
                latest_weight = max(latest_weight, key=lambda f: int(re.search(r'\d+', f.split('step_')[1]).group()))

                final_path = os.path.join(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{seed}", latest_weight.split('.')[0])

                self.models[seed].load(path = final_path, env = self.train_env)

            

            
        for seed, model in self.models.items():

            #NOTE: potentially uncomment if needed
            # new_logger = configure(f"./{self.algorithm}_tensorboard/model_{seed}", ["stdout", "tensorboard"])
            # model.set_logger(new_logger)


            reward_callback = RewardValueCallback(env = self.train_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/", csv_log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_rewards/{self.data_config['observation_space']}/seed_{seed}/", train=True)
            eval_reward_callback = RewardValueCallback(env = self.eval_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/", csv_log_dir=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_rewards/{self.data_config['observation_space']}/seed_{seed}/", train=False)
            gif_callback = GifLoggingCallback(env = self.train_env, save_freq = self.model_config['gif_log_freq'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.data_config['observation_space']}/seed_{seed}/", name_prefix = 'policy_gif')
            checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.data_config['observation_space']}/seed_{seed}/", name_prefix=f'{self.algorithm}_seed{seed}_step', save_replay_buffer=True)

            # Create the callback list
            callback = CallbackList([reward_callback, eval_reward_callback, gif_callback, checkpoint_callback])

            model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{seed}', progress_bar = True, reset_num_timesteps=False, callback = callback)



    def evaluate_policy(self, env, seed, num_timestamps=10):
        all_rewards = []
        for _ in range(num_timestamps):
            obs, info = env.reset()
            episode_reward = 0
            while True:
                action, _states = self.models[seed].predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            all_rewards.append(episode_reward)
        return np.mean(all_rewards)
    
    # TODO Jazlyn
    def tune_hyperparameter():
        pass


def gen_env(seed):
    env = DataGenerator('config.yaml')
    env.reset(seed=seed)
    return env

if __name__ == '__main__':
    print(DataGenerator('config.yaml').observation_space)
    print(DataGenerator('config_test.yaml').observation_space)
    policy_head = PolicyHead(
        SubprocVecEnv([lambda: gen_env(i) for i in range(4)]), 
        DataGenerator('config.yaml'), 
        DataGenerator('config_test.yaml'), 
        'configs/models/config.yaml', 
        'configs/data_generator/config.yaml'
    )
    policy_head.train_and_evaluate_policy(total_timestamps=1000000)



def train_and_evaluate_policy_old(self, total_timestamps):
    train_interval = self.model_config['train_interval']
    eval_interval = self.model_config['eval_interval']
    iteration = 0

    while iteration * (train_interval + eval_interval) < total_timestamps:
        train_rewards = []
        eval_rewards = []

        for seed, model in self.models.items():

            #NOTE: potentially uncomment if needed

            reward_callback = RewardValueCallback(env = self.train_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./logs/{self.algorithm}_tensorboard/{self.data_config['observation_space']}/seed_{seed}/")
            gif_callback = GifLoggingCallback(env = self.train_env, save_freq = self.model_config['gif_log_freq'], log_dir = f"./logs/{self.algorithm}_policyviz/{self.data_config['observation_space']}/seed_{seed}/", name_prefix = 'policy_gif')
            # model.learn(total_timesteps=train_interval, callback=callback)
            # Create the callback list
            callback = CallbackList([reward_callback, gif_callback])
            #progress_bar=True
            model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{seed}', progress_bar = True, reset_num_timesteps=False, callback = callback)
            #TODO: model.save with save dir

            train_reward = self.evaluate_policy(self.train_env, seed, train_interval)
            train_rewards.append(train_reward)

            if (iteration + 1) * (train_interval + eval_interval) <= total_timestamps:
                eval_reward = self.evaluate_policy(self.eval_env, seed, eval_interval)
                eval_rewards.append(eval_reward)

        avg_train_reward = np.mean(train_rewards)
        std_train_reward = np.std(train_rewards)
        avg_eval_reward = np.mean(eval_rewards)
        std_eval_reward = np.std(eval_rewards)

        print(f"Iteration {iteration}:")
        print(f"  Train - Avg Reward: {avg_train_reward}, Std Reward: {std_train_reward}")
        print(f"  Eval  - Avg Reward: {avg_eval_reward}, Std Reward: {std_eval_reward}")

        iteration += 1

