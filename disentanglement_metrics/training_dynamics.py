parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))


from models.ssl_models.custom_supervised import Supervised
from models.ssl_models.custom_barlow_twins import BarlowTwins
from models.ssl_models.factored_models import CovarianceFactorization, MaskingFactorization

from data.load_factored_model import load_factored_model

from omegaconf import DictConfig

from dataset import CustomDataset
from torch.utils.data import DataLoader
import yaml
from collections.abc import Iterable
from scipy.stats import entropy

#number of bins for mutual information computation
NUM_BINS = 999

model_dict = {
    "Supervised": Supervised,
    "BarlowTwins": BarlowTwins,
    "CovarianceFactorization": CovarianceFactorization,
    "MaskingFactorization": MaskingFactorization,
}


def compute_cosine_similarity(tensor1, tensor2):
    # Normalize each tensor to unit length (L2 normalization)
    tensor1_norm = F.normalize(tensor1, p=2, dim=1)
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)

    # Compute cosine similarity for each pair of vectors
    cosine_similarities = torch.sum(tensor1_norm * tensor2_norm, dim=1)

    # Calculate the average cosine similarity
    average_cosine_similarity = cosine_similarities.mean()

    return average_cosine_similarity

def run_training_dynamics(config_path:str='../configs.data_generator/config.yaml', dataset_file:str = ''):
    
    with open(os.path.join(os.path.abspath(__file__), config_path)) as file:
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
        eval_data_path = os.path.relpath('../eval_data/evaluation_samples/factor_samples.pkl')
        test_dataset = CustomDataset(data_env_config=data_configs, limit=None, dataset_file=eval_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    except Exception as e:
        print("ERROR: {}".format(e))
        raise e

    for item in test_dataloader:
        (obs, new_obs, state, new_state, changed_factor) = item

        #concatenate original and changed observations
        all_obs = torch.cat((obs, new_obs), dim=0)
        #preprocessing for factored model forward pass
        all_obs = torch.tensor(all_obs, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

        #re-initialize activation dicitonary before getting hooks for new datapoint
        activation_dict = {}

        register_hooks(model, activations_dict)

        #forward pass data through model to get hooks
        __ = model(all_obs)

        #store the states 
        expert_states_pre.append([item for key in sorted(state.keys()) for item in (state[key] if isinstance(state[key], Iterable) else [state[key]])])
        expert_states_post.append([item for key in sorted(new_state.keys()) for item in (new_state[key] if isinstance(new_state[key], Iterable) else [new_state[key]])])
    

        for layer_name in activation_dict:

            if layer_name not in backbone_activations:
                backbone_activations[layer_name] = [[],[]]
            
            #normalizing activations in batch between 0 to 1
            activation_dict[layer_name] = (activation_dict[layer_name] - activation_dict[layer_name].min()) / (activation_dict[layer_name].max() - activation_dict[layer_name].min())

            if len(activation_dict[layer_name].shape) > 2:
                activation_dict[layer_name] = activation_dict[layer_name].view(activation_dict[layer_name].shape[0], -1)

            split_index = (activation_dict[layer]//2)-1
            backbone_activations[layer_name][0].append(activation_dict[layer_name][:split_index,:])
            backbone_activations[layer_name][1].append(activation_dict[layer_name][split_index:,:])
        


    for layer_name in activation_dict:
        backbone_activations[layer_name] = torch.cat((backbone_activations[layer_name][0], backbone_activations[layer_name][1]), dim=0)
    
    #accumulate and normalize expert states
    expert_states = np.array(expert_states_pre + expert_states_post)
    expert_states = (expert_states - np.min(expert_states, axis=1))/(np.max(expert_states, axis=1)-np.min(expert_states,axis=1))

    expert_states_post = []; expert_states_post =[]
    '''
        USE BACKBONE ACTIVATIONS TO MEASURE METRICS BELOW
    '''
    cosine_similarities = []
    mutual_information_gaps = []
    mutual_informations = []

    for layer in activation_dict:
        

        #metric 1: cosine similarity between flattened representations for each layer
        split_index = (backbone_activations[layer]//2)-1

        half1 = backbone_activations[layer][:split_index]
        half2 = backbone_activations[layer][split_index:]

        avg_cosine_sim = compute_cosine_similarity(half1, half2)
        cosine_similarities.append(avg_cosine_sim)


        #metric 2: mutual information between factors in latent rep
        mutual_information = 0

        for i in range(backbone_activations[layer].shape[1]):
            for j in range(backbone_activations[layer].shape[1]):

                factor1 = backbone_activations[layer][:,i]
                factor2 = backbone_activations[layer][:,j]

                #compute MI between factor1 and factor2
                factor1 = torch.floor(factor1*NUM_BINS)
                factor2 = torch.floor(factor2*NUM_BINS)


                for a in range(NUM_BINS):
                    for b in range(NUM_BINS):

                        p_a_b = (torch.where(factor1==a)&torch.where(factor2==b))[1].shape[0] / backbone_activations[layer].shape[0]

                        p_a = (torch.where(factor1==a))[1].shape[0] / backbone_activations[layer].shape[0]
                        p_b = (torch.where(factor2==b))[1].shape[0] / backbone_activations[layer].shape[0]

                        mutual_information += p_a_b * torch.log(p_a_b/(p_a*p_b))
        

        mutual_informations.append(mutual_information)


        #metric 3: mutual information gap between known factors and latent rep
        for factor in range(expert_states.shape[1]):

            #compute factor's entropy
            values, counts = np.unique(expert_states[:,factor], return_counts=True)
            probabilities = counts / counts.sum()

            factor_entropy = entropy(probabilities, base=2)

            #measure max and second max mutual information 
            max_MI = float('-inf'); second_max_MI = float('-inf')

            for code in range(backbone_activations[layer].shape[1]):

                factors = expert_states[:,factor]
                codes = backbone_activations[layer][:,code]

                #compute MI between factor1 and factor2
                factors = torch.floor(factors*NUM_BINS)
                codes = torch.floor(codes*NUM_BINS)
                
                mutual_information = 0

                for a in range(NUM_BINS):
                    for b in range(NUM_BINS):

                        p_a_b = (torch.where(factors==a)&torch.where(codes==b))[1].shape[0] / backbone_activations[layer].shape[0]

                        p_a = (torch.where(factors==a))[1].shape[0] / backbone_activations[layer].shape[0]
                        p_b = (torch.where(codes==b))[1].shape[0] / backbone_activations[layer].shape[0]

                        mutual_information += p_a_b * torch.log(p_a_b/(p_a*p_b))
                

                #update the max mutual information references
                if mutual_information > max_MI:
                    second_max_MI = max_MI
                    max_MI = mutual_information
                elif mutual_information > second_max_MI:
                    second_max_MI = mutual_information
        
        mutual_information_gaps.append(max_MI - second_max_MI)
     

        
    return cosine_similarities, mutual_informations, mutual_information_gaps




if __name__ == __main__():

    cosine_sim, mutial_info, mutual_info_gaps = run_training_dynamics()


























