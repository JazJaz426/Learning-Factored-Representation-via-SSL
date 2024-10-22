import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.data_generator import DataGenerator
import pdb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import imageio
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch

class RewardValueCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, verbose=0, max_steps: int = 500):
        super(RewardValueCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.max_steps = max_steps
        self.env = env
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        # Save a GIF every save_freq steps
        
        if self.n_calls % self.save_freq == 0:
            # pdb.set_trace()
            self._log_rewards_and_value()
        return True

    
    def _log_rewards_and_value(self):
        # Reset the environment
        obs, info = self.env.reset()
        done = False
        step = 0
        cumulative_reward = 0

        while not done and step < self.max_steps:
            # Predict action and get value function (self.model.predict returns the action and value)
            action,_states = self.model.predict(obs, deterministic=True)
            obs, reward, done, __, info = self.env.step(action)

            # Log the value function and reward to TensorBoard
            cumulative_reward += reward
            self.writer.add_scalar("reward per step", reward, self.num_timesteps + step)
            # self.writer.add_scalar("value per step", value, self.num_timesteps + step)

            step += 1

        # Log the cumulative reward
        self.writer.add_scalar("cumul reward", cumulative_reward, self.num_timesteps)
        self.writer.add_scalar("avg reward", cumulative_reward/step, self.num_timesteps)

        return True
    
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
            #self._log_gif_to_tensorboard(gif_path)
        return True

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
    def __init__(self, train_env, eval_env, model_config_path, data_config_path):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()
        self.train_env = train_env
        self.eval_env = eval_env
        # pdb.set_trace()
        self.models = self.create_models(num_models=self.model_config['num_models'])

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
                ppo_params = {k: v for k, v in self.model_config['ppo'].items() if (v is not None and v is True)}
                
                models[model_num] = PPO(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    verbose = self.model_config['verbose'],
                    **ppo_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/seed_{model_num}/"
                )
            elif self.algorithm == "DQN":
                dqn_params = {k: v for k, v in self.model_config['dqn'].items() if (v is not None and v is True)}
                models[model_num] = DQN(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    verbose = self.model_config['verbose'],
                    **dqn_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/seed_{model_num}/"
                )
            elif self.algorithm == "A2C":
                a2c_params = {k: v for k, v in self.model_config['a2c'].items() if (v is not None and v is True)}
                models[model_num] = A2C(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    verbose= self.model_config['verbose'],
                    **a2c_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/seed_{model_num}/"
                )
            else:
                raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        return models

    def train_and_evaluate_policy(self, total_timestamps):
        train_interval = self.model_config['train_interval']
        eval_interval = self.model_config['eval_interval']
        iteration = 0

        while iteration * (train_interval + eval_interval) < total_timestamps:
            train_rewards = []
            eval_rewards = []

            for seed, model in self.models.items():

                #NOTE: potentially uncomment if needed
                # new_logger = configure(f"./{self.algorithm}_tensorboard/model_{seed}", ["stdout", "tensorboard"])
                # model.set_logger(new_logger)

                reward_callback = RewardValueCallback(env = self.train_env, save_freq = self.model_config['reward_log_freq'], log_dir=f"./{self.algorithm}_tensorboard/seed_{seed}/")
                gif_callback = GifLoggingCallback(env = self.train_env, save_freq = self.model_config['gif_log_freq'], log_dir = f"./{self.algorithm}_policyviz/seed_{seed}/", name_prefix = 'policy_gif')
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

if __name__ == '__main__':
    policy_head = PolicyHead(DataGenerator(), DataGenerator(), 'configs/models/config.yaml', 'configs/data_generator/config.yaml')
    policy_head.train_and_evaluate_policy(total_timestamps=100000)