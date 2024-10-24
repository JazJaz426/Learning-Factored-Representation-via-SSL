import os
from PIL import Image
import torch

from tqdm import tqdm
import numpy as np
import copy

action_list = ["left","right","forward","pickup","drop","activate","done"]

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
        self._set_data_generator_reset_type("default")

        do_random_policy = True if policy_net == None else False

        samples = []
        for episode in tqdm(range(num_episodes), desc="Running episodes"):

            episode_dict = {"obs":[], "info":[], "action":[], "reward":[]}
            done = False
            steps = 0
            obs, info = self._get_reset_data_generator()
            episode_dict["obs"].append(obs)   # obs:  image, expert, or factored
            episode_dict["info"].append(info) # info: everything, regard as learning target for obs

            while not done and steps<self.episode_max_length:
                if do_random_policy:
                    action = self._get_data_generator_random_action()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits = policy(obs_tensor)
                    action = torch.distributions.Categorical(logits=logits).sample()
                obs, reward, done, _, info = self._get_step_data_generator(action)
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
    def sample_obs_pairs(self, num_pairs_per_control_attr):
        print("  Sampling observation pairs...")
        obs_pairs_root_path = os.path.join(self.dataset_root_path, "obs_pairs")
        if not os.path.exists(obs_pairs_root_path):
            os.makedirs(obs_pairs_root_path)

        self._set_data_generator_reset_type("custom")

        for control_attr in tqdm(self._get_data_generator_state_attr(), desc="Running observation pairs"):
            control_attr_save_path = os.path.join(obs_pairs_root_path, control_attr)
            if not os.path.exists(control_attr_save_path):
                os.makedirs(control_attr_save_path)
            
            i=0
            while i < num_pairs_per_control_attr:
                control_attr_sample = self._gen_rand_control_attr(control_attr)
                self._set_data_generator_control_attr({control_attr:control_attr_sample})
                try:
                    obs0, info0 = self._get_reset_data_generator()
                except RecursionError as e:
                    print(f"RecursionError occurred: {e}")
                    continue
                try:
                    obs1, info1 = self._get_reset_data_generator()
                except RecursionError as e:
                    print(f"RecursionError occurred: {e}")
                    continue

                self._save_obs_pairs(obs0, obs1, info0['state_dict'], info1['state_dict'], control_attr_save_path, i)
                i+=1
                

    # triplet (o,o',o'',a',a'')
    def sample_obs_triplets(self, num_triplets):
        print("  Sampling observation triplets...")
        obs_triplets_root_path = os.path.join(self.dataset_root_path, "obs_triplets")
        if not os.path.exists(obs_triplets_root_path):
            os.makedirs(obs_triplets_root_path)

        for i in tqdm(range(num_triplets),desc="Running observation triplets"):
            obs_triplets_idx_path = os.path.join(obs_triplets_root_path, str(i))
            if not os.path.exists(obs_triplets_idx_path):
                os.makedirs(obs_triplets_idx_path)

            self._set_data_generator_reset_type("custom")
            self._set_data_generator_control_attr({})
            init_obs, init_info = self._get_reset_data_generator()
            temp_data_generator = self._get_copy_data_generator()

            # Assume do random policy
            action = self._get_data_generator_random_action()
            obs, _, _, _, info = self._get_step_data_generator(action)

            #self._set_data_generator_reset_type("custom")
            #self._set_data_generator_control_attr(init_info["state_dict"])
            #init_obs1, _ = self._get_reset_data_generator()

            # action_prime = self._get_data_generator_random_action()
            action_prime = temp_data_generator.action_space.sample()
            while action_prime == action:
                # action_prime = self._get_data_generator_random_action()
                action_prime = temp_data_generator.action_space.sample()
            obs_prime, _, _, _, info_prime = temp_data_generator.step(action_prime)

            self._save_obs_triplets(init_obs, obs, obs_prime, action, action_prime, obs_triplets_idx_path,
                                    init_state_dict=init_info["state_dict"], 
                                    state_dict0=info["state_dict"], 
                                    state_dict1=info_prime["state_dict"]
                                    )

    def _set_data_generator_reset_type(self, reset_type):
        assert reset_type in ["custom", "random", "default"]
        self.data_generator.reset_type = reset_type

    def _get_reset_data_generator(self):
        return self.data_generator.reset()

    def _get_copy_data_generator(self):
        return copy.deepcopy(self.data_generator)

    def _set_data_generator_control_attr(self, control_attr_dict):
        self.data_generator.controlled_factors = control_attr_dict

    def _get_step_data_generator(self, action):
        return self.data_generator.step(action)
    
    def _get_data_generator_random_action(self):
        return self.data_generator.action_space.sample()

    def _get_data_generator_state_attr(self):
        return self.data_generator.state_attributes

    def _gen_rand_control_attr(self, control_attr):
        gym_space_params = self.data_generator.gym_space_params
        attribute_types = self.data_generator.state_attribute_types[control_attr]

        sample_val = []
        for attr_type in attribute_types:
            min_val, max_val, dtype = gym_space_params[attr_type]
            if attr_type == "coordinate_width" or attr_type == "coordinate_height":
                min_val = min_val + 1
                max_val = max_val - 1
            elif attr_type == "boolean":
                min_val = 0
                max_val = 2

            random_val = np.random.randint(min_val, max_val)
            sample_val.append(dtype(random_val))

        if len(sample_val)==1:
            return sample_val[0]
        else:
            return np.array(sample_val)
    
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

    def _save_obs_pairs(self, obs0, obs1, state_dict0, state_dict1, save_path, idx):
        observation_type = self.data_generator.observation_type
        if observation_type == 'image':
            image_path = os.path.join(save_path, f"obs-image-{idx}-0.png")
            img = Image.fromarray(obs0)
            img.save(image_path)
            image_path = os.path.join(save_path, f"obs-image-{idx}-1.png")
            img = Image.fromarray(obs1)
            img.save(image_path)
        elif observation_type == 'expert':
            expert_path = os.path.join(save_path, f"obs-expert-{idx}-0.npy")
            expert_array = np.array(obs0)
            np.save(expert_path, expert_array)     
            expert_path = os.path.join(save_path, f"obs-expert-{idx}-1.npy")
            expert_array = np.array(obs1)
            np.save(expert_path, expert_array)  
        elif observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')
        else:
            raise Exception('ERROR: observation type {} undefined'.format(observation_type))
        state_dict_array = np.array(state_dict0)
        np.save(os.path.join(save_path,f"state_dict-{idx}-0.npy"), state_dict_array)  
        state_dict_array = np.array(state_dict1)
        np.save(os.path.join(save_path,f"state_dict-{idx}-1.npy"), state_dict_array)  

    def _save_obs_triplets(self, init_obs, obs0, obs1, action0, action1, save_path,
                           init_state_dict=None, state_dict0=None, state_dict1=None):
        observation_type = self.data_generator.observation_type
        if observation_type == 'image':
            image_path = os.path.join(save_path, "obs-image-init.png")
            img = Image.fromarray(init_obs)
            img.save(image_path)
            image_path = os.path.join(save_path, f"obs-image-{action_list[int(action0)]}-0.png")
            img = Image.fromarray(obs0)
            img.save(image_path)
            image_path = os.path.join(save_path, f"obs-image-{action_list[int(action1)]}-1.png")
            img = Image.fromarray(obs1)
            img.save(image_path)
        elif observation_type == 'expert':
            expert_path = os.path.join(save_path, "obs-expert-init.npy")
            expert_array = np.array(init_obs)
            np.save(expert_path, expert_array)     
            expert_path = os.path.join(save_path, f"obs-expert-{action_list[int(action0)]}-0.npy")
            expert_array = np.array(obs0)
            np.save(expert_path, expert_array) 
            expert_path = os.path.join(save_path, f"obs-expert-{action_list[int(action1)]}-1.npy")
            expert_array = np.array(obs1)
            np.save(expert_path, expert_array)  
        elif observation_type == 'factored':
            raise NotImplementedError('ERROR: to be implemented after factored representation encoder')
        else:
            raise Exception('ERROR: observation type {} undefined'.format(observation_type))
        action_path = os.path.join(save_path, "action.npy")
        action_array = np.array([action0, action1])
        np.save(action_path, action_array) 

        if init_state_dict:
            state_dict_array = np.array(init_state_dict)
            np.save(os.path.join(save_path,f"state_dict-init.npy"), state_dict_array) 
        if state_dict0:
            state_dict_array = np.array(state_dict0)
            np.save(os.path.join(save_path,f"state_dict-0.npy"), state_dict_array) 
        if state_dict1:
            state_dict_array = np.array(state_dict1)
            np.save(os.path.join(save_path,f"state_dict-1.npy"), state_dict_array) 

if __name__ == '__main__':
    from data.data_generator import DataGenerator

    data_generator = DataGenerator(config_path='./configs/data_generator/config.yaml')
    mdg = GenerateDataset(data_generator, dataset_root_path='./temp_samples_3', episode_max_length=10)

    mdg.sample_episodes(num_episodes=30)
    mdg.sample_obs_pairs(num_pairs_per_control_attr=30)
    mdg.sample_obs_triplets(num_triplets=30)