import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.data_generator import DataGenerator
import pdb
from stable_baselines3.common.callbacks import BaseCallback


class RewardValueCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardValueCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        self.logger.record('reward', np.mean(reward))

        value = self.model.policy.predict_values(self.locals['obs'])
        self.logger.record('value_function', np.mean(value))

        return True


class PolicyHead:
    def __init__(self, train_env, eval_env, model_config_path, data_config_path):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()
        self.train_env = train_env
        self.eval_env = eval_env
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
                ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}
                models[model_num] = PPO(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **ppo_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/"
                )
            elif self.algorithm == "DQN":
                dqn_params = {k: v for k, v in self.model_config['dqn'].items() if v is not None}
                models[model_num] = DQN(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **dqn_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/"
                )
            elif self.algorithm == "A2C":
                a2c_params = {k: v for k, v in self.model_config['a2c'].items() if v is not None}
                models[model_num] = A2C(
                    policy=self.policy_name,
                    env=self.train_env,
                    seed=model_num,
                    **a2c_params,
                    tensorboard_log=f"./{self.algorithm}_tensorboard/"
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
                new_logger = configure(f"./{self.algorithm}_tensorboard/model_{seed}", ["stdout", "tensorboard"])
                model.set_logger(new_logger)

                callback = RewardValueCallback()
                model.learn(total_timesteps=train_interval, callback=callback)

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
    pdb.set_trace()
    policy_head = PolicyHead(DataGenerator(), DataGenerator(), 'configs/models/config.yaml', 'configs/data_generator/config.yaml')
    policy_head.train_and_evaluate_policy(total_timestamps=10000)