"""
some copied from EvalCallback. modified by waymao
"""
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from vec_video_recorder import VecVideoRecorder

import numpy as np
from gymnasium import error, logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from typing import Callable, List
from matplotlib import pyplot as plt
import gymnasium as gym
import pandas as pd 
import pdb

#NOTE: for older RewardValueCallback used as global variable building CSV
csv_logger = pd.DataFrame(columns=['train/test', 'step', 'seed', 'cumul_reward'])

class CustomVideoRecorder(VecVideoRecorder):

    """
    Custom Video Recorder wrapper for SB3 with following extra features:
    (1) stops video when policy is done 
    """
    
    def __init__(self, venv: VecEnv, video_folder: str, record_video_trigger: Callable[[int], bool], video_length: int = 200, name_prefix: str = "rl-video"):
        super(CustomVideoRecorder, self).__init__(venv, video_folder, record_video_trigger, video_length, name_prefix)

        #[SR] extra variable to track when different environments are "done"
        self.dones = np.full((venv.num_envs,), False)

        #[SR] extra variable to track last observation: to preserve in case some envs are "done" before others
        self.terminal_obs = [None for _ in range(venv.num_envs)]
    
 
    def step_wait(self) -> VecEnvStepReturn:
        
        
        obs, rewards, dones, infos = self.env.step_wait()

        #[SR] boolean or to set self.dones true: stop recording when all envs reach done
        # self.dones = np.logical_or(self.dones, dones)

        # #[SR] in case some dones are True, store in self.terminal_obs and reset the obs
        # for i, done in enumerate(dones):

        #     if done:
        #         self.terminal_obs[i] = obs[i,:,:,:]
        
        # for i in range(len(obs)):

        #     if isinstance(self.terminal_obs[i], np.ndarray):
        #         obs[i,:,:,:] = self.terminal_obs[i]

        self.step_id += 1
        if self.recording:

            self._capture_frame()
            if (self.dones.all() == True) or (len(self.recorded_frames) > self.video_length):
                print(f"Saving video to {self.video_path}")
                self._stop_recording()
                
        elif self._video_enabled():
            self._start_video_recorder()

        return obs, rewards, dones, infos

class CustomEvalCallback(EvalCallback):
    """
    Custom Evaluation Callback for Stable Baselines that supports custom
    name for logging.
    """
    def __init__(self, custom_name, eval_env, gamma=None, *args, **kwargs):
        #TODO: chekc kwargs input into model, correctly passed to super() EvalCallback
        self.custom_name = custom_name
        self.gamma = gamma
        super(CustomEvalCallback, self).__init__(eval_env, *args, **kwargs)

        self.mean_evaluations_results = []
        self.std_evaluations_results = []
        self.mean_evaluations_length = []
        self.std_evaluations_length = []

    def _on_step(self) -> bool:
        # NOTE: this function is identical to the original _on_step function
        # but with the addition of the custom name for logging
        continue_training = True

        #[SR] divided by self.eval_env.num_envs for periodic logging
        if self.eval_freq > 0 and (self.n_calls % (self.eval_freq // self.eval_env.num_envs))== 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.mean_evaluations_results.append(np.mean(episode_rewards))
                self.std_evaluations_results.append(np.std(episode_rewards))
                self.mean_evaluations_length.append(np.mean(episode_lengths))
                self.std_evaluations_length.append(np.std(episode_lengths))

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    mean_results = self.mean_evaluations_results,
                    std_results = self.std_evaluations_results,
                    ep_lengths=self.evaluations_length,
                    mean_ep_lengths = self.mean_evaluations_length,
                    std_ep_lengths = self.std_evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            if self.gamma is not None:
                mean_discounted_return = np.mean(episode_rewards * np.power(self.gamma, episode_lengths))
                std_discounted_return = np.std(episode_rewards * np.power(self.gamma, episode_lengths))
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print("=----------------------------=")
                print(f"{self.custom_name}: num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/{self.custom_name}/mean_reward", float(mean_reward))
            self.logger.record(f"eval/{self.custom_name}/std_reward", float(std_reward))
            self.logger.record(f"eval/{self.custom_name}/mean_ep_length", mean_ep_length)
            self.logger.record(f"eval/{self.custom_name}/std_ep_length", std_ep_length)
            if self.gamma is not None:
                self.logger.record(f"eval/{self.custom_name}/mean_discounted_return", float(mean_discounted_return))
                self.logger.record(f"eval/{self.custom_name}/std_discounted_return", float(std_discounted_return))



            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                    print("=----------------------------=")

                self.logger.record(f"eval/{self.custom_name}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/{self.custom_name}/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class ValuePlottingCallback(BaseCallback):

    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, num_envs: int, name_prefix: str = "", verbose=1):
        super(ValuePlottingCallback, self).__init__(verbose)

        #[SR] dummy base env to generate value function
        self.base_env = env
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.name_prefix = name_prefix


        #[SR] add separate video folder for value logging
        self.value_folder = os.path.join(log_dir, 'value_functions')
        os.makedirs(self.value_folder, exist_ok=True)

        

        self.num_envs = num_envs
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        
        # Save a GIF every save_freq steps
        if self.n_calls % (self.save_freq//self.num_envs) == 0:
            self._plot_value_func()
            
        return True
   
    def _plot_value_func(self):
        '''extract the value function for a particular observation'''
        
        original_obs, info = self.base_env.reset()

        value_function = np.zeros((self.base_env.env.env.unwrapped.width, self.base_env.env.env.unwrapped.height)) * np.nan


        for w in range(self.base_env.env.env.unwrapped.width):
            for h in range(self.base_env.env.env.unwrapped.height):

                obj = self.base_env.env.env.unwrapped.grid.get(w, h)

                if obj is None:

                    # place the agent here
                    self.base_env.env.env.unwrapped.agent_pos = (w, h)

                    value_estim = []
                    
                    for dir in range(4):
                        
                        self.base_env.env.env.unwrapped.agent_dir = dir

                        obs = self.base_env.get_curr_obs()
                        
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
        plt.imshow(info['obs'])
        plot_extent = [0, self.base_env.env.env.unwrapped.width * 8, self.base_env.env.env.unwrapped.height * 8, 0]
        plt.imshow(value_function, alpha=0.5, cmap='viridis', extent = plot_extent)
        plt.colorbar()
        plt.title(f"Value Function @ Step {self.num_timesteps}")


        value_name = f"{self.name_prefix}-step-{self.num_timesteps}.png"
        value_path = os.path.join(self.value_folder, value_name)

        plt.savefig(value_path)
        plt.close()

    def _on_training_end(self):
        plt.close()



#NOTE: RewardValueCallback is not used

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
