import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))


from models.ssl_models.custom_supervised import Supervised
from models.ssl_models.custom_barlow_twins import BarlowTwins
from models.ssl_models.factored_models import CovarianceFactorization, MaskingFactorization

from data.load_factored_model import load_factored_model

from omegaconf import DictConfig

from data.dataset import CustomDataset
from torch.utils.data import DataLoader
import yaml
from collections.abc import Iterable
from scipy.stats import entropy
import pdb
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

#number of bins for mutual information computation
NUM_BINS = 999
#epsilon to avoid division by zero
EPSILON = 1e-8

model_dict = {
    "Supervised": Supervised,
    "BarlowTwins": BarlowTwins,
    "CovarianceFactorization": CovarianceFactorization,
    "MaskingFactorization": MaskingFactorization,
}


def compute_cosine_similarity(tensor1, tensor2):
    # Normalize each tensor to unit length (L2 normalization)
    tensor1_norm = nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2_norm = nn.functional.normalize(tensor2, p=2, dim=1)

    # Compute cosine similarity for each pair of vectors
    cosine_similarities = torch.sum(tensor1_norm * tensor2_norm, dim=1)

    # Calculate the average cosine similarity
    average_cosine_similarity = cosine_similarities.mean()

    return average_cosine_similarity

def compute_mutual_information(x, y, bins=10):
    """
    Calculates mutual information between two discrete random variables without using nested for loops.
    """
    # Joint probability (2D histogram)
    p_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)

    # Marginal probabilities
    p_x = np.histogram(x, bins=x_edges, density=True)[0]
    p_y = np.histogram(y, bins=y_edges, density=True)[0]

    # Avoiding log(0) by setting 0 values to a small positive number
    p_xy = np.where(p_xy > 0, p_xy, 1e-10)
    p_x = np.where(p_x > 0, p_x, 1e-10)
    p_y = np.where(p_y > 0, p_y, 1e-10)

    # Calculate mutual information
    
    mi = np.sum(p_xy * np.log2(p_xy / np.outer(p_x, p_y)))

    return mi

def run_training_dynamics(config_path:str='../configs/data_generator/config.yaml', dataset_file:str = '', compute_mi=True, compute_mig=True):
    
    # pdb.set_trace()

    with open(os.path.normpath(config_path)) as file:
        data_configs = yaml.safe_load(file)

    
    #get the model being used, load its weights from checkpoint file and set backbone to eval mode
    try:
        model = load_factored_model(
            data_configs['factored_model']['model'], 
            data_configs['factored_model']['checkpoint']
        )
    except Exception as e:
        print("Unable to load factorized model")
        raise e

    model.eval()


    #setup model hooks to get intermediate representations and expert states
    backbone_activations = {}
    expert_states_pre = []
    expert_states_post = []

    def hook_fn(name, activations_dict):
        def hook(module, input, output):
            activations_dict[name] = output.detach().cpu()
        return hook
    

    def register_hooks(model, activations_dict):
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.Linear):
                layer.register_forward_hook(hook_fn(name, activations_dict))

    

    try:
        eval_data_path = os.path.relpath('../data/eval_data/evaluation_samples/factor_samples.pkl')
        test_dataset = CustomDataset(data_env_config=config_path, limit=None, dataset_file=eval_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    except Exception as e:
        print("ERROR: {}".format(e))
        raise e

    
    for item in test_dataloader:
        (obs, new_obs, state, new_state, changed_factor) = item

        #concatenate original and changed observations
        all_obs = torch.cat((obs, new_obs), dim=0)
        #preprocessing for factored model forward pass
        all_obs = all_obs.to(dtype=torch.float32)
        all_obs = all_obs.permute(0,3,1,2)

        #re-initialize activation dicitonary before getting hooks for new datapoint
        activation_dict = {}

        register_hooks(model, activation_dict)

        
        #forward pass data through model to get hooks
        __ = model(all_obs)

        #store the states 
        expert_states_pre.append( torch.stack([item for key in sorted(state.keys()) for item in (state[key] if isinstance(state[key], list) else [state[key]])],  dim=1) )
        expert_states_post.append( torch.stack([item for key in sorted(new_state.keys()) for item in (new_state[key] if isinstance(new_state[key], list) else [new_state[key]])], dim=1) )
    
        
        for layer_name in activation_dict:

            if layer_name not in backbone_activations:
                backbone_activations[layer_name] = [[],[]]
            
            if len(activation_dict[layer_name].shape) > 2:
                activation_dict[layer_name] = activation_dict[layer_name].reshape(activation_dict[layer_name].shape[0], -1)

            split_index = (activation_dict[layer_name].shape[0]//2)
            backbone_activations[layer_name][0].append(activation_dict[layer_name][:split_index,:])
            backbone_activations[layer_name][1].append(activation_dict[layer_name][split_index:,:])


            

    # pdb.set_trace()
    for layer_name in activation_dict:
        backbone_activations[layer_name][0] = torch.cat(backbone_activations[layer_name][0], axis=0)
        backbone_activations[layer_name][1] = torch.cat(backbone_activations[layer_name][1], axis=0)
        backbone_activations[layer_name] = torch.cat((backbone_activations[layer_name][0], backbone_activations[layer_name][1]), dim=0)
        
        #normalizing activations in batch between 0 to 1
        max_vals, _ = torch.max(backbone_activations[layer_name], dim=1, keepdim=True)
        min_vals, _ = torch.min(backbone_activations[layer_name], dim=1, keepdim=True)

        # Compute the denominator safely, avoiding division by zero
        denominator = torch.where((max_vals - min_vals) == 0, torch.ones_like(max_vals), (max_vals - min_vals))
        
        backbone_activations[layer_name] = (backbone_activations[layer_name] - min_vals) / denominator
        
        print(f"Max: {backbone_activations[layer_name].max()}, Min: {backbone_activations[layer_name].min()}")

    
    #accumulate and normalize expert states
    expert_states_pre = torch.cat(expert_states_pre, axis=0)
    expert_states_post = torch.cat(expert_states_post, axis=0)

    expert_states = torch.cat([expert_states_pre, expert_states_post], dim=0)

    max_vals, _ = torch.max(expert_states, dim=1, keepdim=True)
    min_vals, _ = torch.min(expert_states, dim=1, keepdim=True)
    denominator = torch.where((max_vals - min_vals) == 0, torch.ones_like(max_vals), (max_vals - min_vals))
    expert_states = (expert_states - min_vals)/denominator

    expert_states_post = []; expert_states_post =[]
    '''
        USE BACKBONE ACTIVATIONS TO MEASURE METRICS BELOW
    '''
    cosine_similarities = []
    mutual_information_gaps = []
    mutual_informations = []
    eucledian_distances = []

    for layer in tqdm(activation_dict):
        

        #metric 1: cosine similarity between flattened representations for each layer
        split_index = (backbone_activations[layer].shape[0]//2)

        half1 = backbone_activations[layer][:split_index]
        half2 = backbone_activations[layer][split_index:]

        avg_cosine_sim = compute_cosine_similarity(half1, half2)
        cosine_similarities.append(avg_cosine_sim)

        avg_euclidean_dist = torch.norm(half1 - half2, dim=1).mean()
        eucledian_distances.append(avg_euclidean_dist)
        


        #metric 2: mutual information gap between known factors and latent rep
        if compute_mig:
            for factor in range(expert_states.shape[1]):

                #compute factor's entropy
                values, counts = np.unique(expert_states[:,factor], return_counts=True)
                probabilities = counts / counts.sum()

                factor_entropy = np.maximum(entropy(probabilities, base=2), EPSILON)

                #measure max and second max mutual information 
                max_MI = float('-inf'); second_max_MI = float('-inf')

                for code in range(backbone_activations[layer].shape[1]):

                    factors = expert_states[:,factor]
                    codes = backbone_activations[layer][:,code]

                    #compute MI between factor1 and factor2
                    factors = torch.floor(factors*NUM_BINS).to(torch.int)
                    codes = torch.floor(codes*NUM_BINS).to(torch.int)

                    mutual_information = mutual_info_score(factors, codes)
                    
                    # mutual_information = 0

                    # for a in range(tqdm(NUM_BINS)):
                    #     for b in range(NUM_BINS):

                    #         p_a_b = (torch.where(factors==a)&torch.where(codes==b))[1].shape[0] / backbone_activations[layer].shape[0]

                    #         p_a = (torch.where(factors==a))[1].shape[0] / backbone_activations[layer].shape[0]
                    #         p_b = (torch.where(codes==b))[1].shape[0] / backbone_activations[layer].shape[0]

                    #         mutual_information += p_a_b * torch.log(p_a_b/(p_a*p_b))
                    

                    #update the max mutual information references
                    if mutual_information > max_MI:
                        second_max_MI = max_MI
                        max_MI = mutual_information
                    elif mutual_information > second_max_MI:
                        second_max_MI = mutual_information
            
                print(f'Finished Factor: {factor}')
            mutual_information_gaps.append((max_MI - second_max_MI)/factor_entropy)

        


        #metric 3: mutual information between factors in latent rep

        if compute_mi:

            mutual_information = 0

            for i in range(backbone_activations[layer].shape[1]):
                for j in range(backbone_activations[layer].shape[1]):
                
                    print(f'Running {i},{j}')


                    factor1 = backbone_activations[layer][:split_index,i]
                    factor2 = backbone_activations[layer][split_index:,j]

                    #compute MI between factor1 and factor2
                    factor1 = torch.floor(factor1*NUM_BINS).to(torch.int)
                    factor2 = torch.floor(factor2*NUM_BINS).to(torch.int)

                    
                    mutual_information += mutual_info_score(factor1, factor2)


                    # for a in range(NUM_BINS):
                    #     for b in range(NUM_BINS):

                    #         p_a_b = (torch.where((factor1==a) & (factor2==b))[0].shape[0]) / backbone_activations[layer].shape[0]

                    #         p_a = (torch.where(factor1==a))[0].shape[0] / backbone_activations[layer].shape[0]
                    #         p_b = (torch.where(factor2==b))[0].shape[0] / backbone_activations[layer].shape[0]

                    #         if p_a!=0 and p_b != 0:
                    #             log_term = p_a_b / (p_a * p_b)
                    #         else:
                    #             log_term = EPSILON

                    #         mutual_information += p_a_b * np.log(log_term)
        

            mutual_informations.append(mutual_information/(backbone_activations[layer].shape[1]*backbone_activations[layer].shape[1]))


        
     

        
    return cosine_similarities, eucledian_distances, mutual_informations, mutual_information_gaps




if __name__ == '__main__':

    cosine_sim, eucledian_distances, mutual_info, mutual_info_gaps = run_training_dynamics(compute_mi=False, compute_mig=True)
    print(f"Cosine Similarities: {cosine_sim}")
    print(f"Euclidean Distances: {eucledian_distances}")
    print(f"Mutual Info Gap: {mutual_info_gaps}")


























