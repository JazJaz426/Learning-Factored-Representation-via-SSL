

import numpy as np
from sklearn.preprocessing import minmax_scale
from pyitlib import discrete_random_variable as drv

def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables
    
    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)

# [ch] https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/mig.py
# [ch] does not pass tests: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils_test.py 
'''
def discrete_entropy(y):
    num_factors = y.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = get_mutual_information(y[j,:], y[j,:])
    return h
'''

# [ch] pass tests: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils_test.py 
def discrete_entropy(ys):
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    
    for j in range(num_factors):
        # Calculate entropy using the probabilities of each class
        prob_distribution = np.bincount(ys[:,j]) / ys.shape[0]  # Probability of each class
        prob_distribution = prob_distribution[prob_distribution > 0]  # Remove zero probabilities
        h[j] = -np.sum(prob_distribution * np.log(prob_distribution))  # Entropy formula
        
    return h

    
# [ch] Usage:
# [ch]      1. **factors, codes, just pass to the function as their raw form, set continuous_factors=True.**
# [ch]      2. nb_bins, batch_size leave as default.
# [ch]      !? weird, entropy
def mig(factors, codes, continuous_factors=True, nb_bins=10):
    ''' MIG metric from R. T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud,
        “Isolating sources of disentanglement in variationalautoencoders,”
        in NeurIPS, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # quantize latent codes
    codes = minmax_scale(codes)  # normalize in [0, 1] all columns
    codes = get_bin_index(codes, nb_bins)  # quantize values and get indexes

    # compute mutual information matrix
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_matrix[f, c] = get_mutual_information(factors[:, f], codes[:, c])

    entropy = discrete_entropy(factors) # !?

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        mi_f = np.sort(mi_matrix[f, :])
        # get diff between highest and second highest term and add it to total gap
        # sum_gap += mi_f[-1] - mi_f[-2] 
        sum_gap += (mi_f[-1] - mi_f[-2]) / entropy[f]
    
    # compute the mean gap
    # mig_score = sum_gap / nb_factors !?
    
    # return mig_score
    return sum_gap

if __name__ == "__main__":
    
    factors = np.random.randint(0, 10,size=(1000, 4))
    codes =  factors+1
    target = np.transpose(np.array([[1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]]))
    print(discrete_entropy(target))
    print(mig(factors,codes))