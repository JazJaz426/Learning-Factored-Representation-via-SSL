import os
from PIL import Image
import torch

from tqdm import tqdm
import numpy as np


class GenerateDataset:
    def __init__(
        self,
        data_generator,
        dataset_root_path,
        episode_max_length
    ):
        self.data_generator         = data_generator
        self.dataset_root_path      = dataset_root_path
        self.episode_max_length     = episode_max_length

        if not os.path.exists(dataset_root_path):
            os.makedirs(dataset_root_path)

        self._save_func = {"episode_obs":       self._save_episode_obs, 
                           "episode_info":      self._save_episode_info,
                           "episode_action":    self._save_episode_action,
                           "episode_reward":    self._save_episode_reward
                            }

    # episode (o0, a0, o1, a1,..., on)
    def sample_episodes(self, policy_net=None, optimizer=None, num_episodes=1000, save_features=["obs","action","info"]):
        print("  Sampling episodes...")
        do_random_policy = True if policy_net == None else False

        samples = []
        for episode in tqdm(range(num_episodes), desc="Running episodes"):

            episode_dict = {"obs":[], "info":[], "action":[], "reward":[]}
            done = False
            steps = 0
            obs, info = self.data_generator.reset()
            episode_dict["obs"].append(obs)   # obs:  image, expert, or factored
            episode_dict["info"].append(info) # info: everything, regard as learning target for obs

            while not done and steps<self.episode_max_length:
                if do_random_policy:
                    action = self.data_generator.action_space.sample()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits = policy(obs_tensor)
                    action = torch.distributions.Categorical(logits=logits).sample()
                obs, reward, done, _, info = data_generator.step(action)
                episode_dict["obs"].append(obs)
                episode_dict["info"].append(info)
                episode_dict["action"].append(action)
                episode_dict["reward"].append(reward)
                steps += 1

            # Add episode to samples
            samples.append(episode_dict)

            if not do_random_policy:
                # TODO Calculate returns (discounted rewards)
                # TODO Calculate policy loss and update policy
                pass
        
        # Save samples
        episodes_root_path = os.path.join(self.dataset_root_path, "episodes")
        if not os.path.exists(episodes_root_path):
            os.makedirs(episodes_root_path)

        for i in tqdm(range(num_episodes), desc="Saving episodes"):
            episode_i_path = os.path.join(episodes_root_path, str(i))
            if not os.path.exists(episode_i_path):
                os.makedirs(episode_i_path)
            
            for feature in save_features:
                save_path = os.path.join(episode_i_path, feature)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self._save_func[f"episode_{feature}"](samples[i][feature], save_path)

    # pair (o,o',factor_k)
    def sample_obs_pairs(self, num_pairs):
        print("  Sampling observation pairs...")
        #options: [agent_pos: [x,y], agent_dir: [int], goal_pos: [x,y], key_pos: [x,y], door_pos: [x,y], holding_key: true/false, door_locked: true/false, door_open: true/false]
        #agent_dir: 0 - right, 1 - down, 2 - left, 3 - up
        print("state_attributes", self.data_generator.state_attributes)

        for control_attributes in self.data_generator.state_attributes:
            for j in range(num_pairs):
                continue

        

    # triplet (o,o',o'',a',a'')
    def sample_obs_triplets(self, num_triplets):
        pass
    
    def _save_episode_obs(self, episode_obs, save_path):
        observation_type = self.data_generator.observation_type
        if observation_type == 'image':
            for i in range(len(episode_obs)):
                image_path = os.path.join(save_path, f"obs-image-{i}.png")
                img = Image.fromarray(episode_obs[i])
                img.save(image_path)
        elif observation_type == 'expert':
            expert_path = os.path.join(save_path, "obs-expert.npy")
            expert_array = np.array(episode_obs)
            np.save(expert_path, expert_array)     
        elif observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')
        else:
            raise Exception('ERROR: observation type {} undefined'.format(observation_type))
    
    def _save_episode_action(self, episode_action, save_path):
        action_path = os.path.join(save_path, "action.npy")
        action_array = np.array(episode_action)
        np.save(action_path, action_array)  
    
    def _save_episode_info(self, episode_info, save_path):
        state_dict_list = []
        for i in range(len(episode_info)):
            for info_key, info_val in episode_info[i].items():
                if info_key != "state_dict":
                    attr_save_path = os.path.join(save_path, info_key)
                    if not os.path.exists(attr_save_path):
                        os.makedirs(attr_save_path)
                    image_path = os.path.join(attr_save_path, f"info-{info_key}-{i}.png")
                    img = Image.fromarray(episode_info[i][info_key])
                    img.save(image_path)
                else:
                    state_dict_list.append(episode_info[i][info_key])

        state_dict_array = np.array(state_dict_list)
        np.save(os.path.join(save_path,"state_dict.npy"), state_dict_array)  

    def _save_episode_reward(self, episode_reward, save_path):
        reward_path = os.path.join(save_path, "reward.npy")
        action_array = np.array(episode_reward)
        np.save(reward_path, action_array) 


if __name__ == '__main__':
    from data.data_generator import DataGenerator

    data_generator = DataGenerator(config_path='./configs/data_generator/config.yaml')
    mdg = GenerateDataset(data_generator, dataset_root_path='./temp_samples', episode_max_length=10)

    #mdg.sample_episodes(num_episodes=5)
    mdg.sample_obs_pairs(num_pairs=5)
    #mdg.sample_obs_triplets(num_triplets=5)