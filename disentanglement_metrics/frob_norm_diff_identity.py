import numpy as np
from scipy.stats import spearmanr

def frob_norm_diff_identity(codes, corr_type="correlation"):
    """
    Computes the Frobenius norm of the correlation or Spearman correlation matrix 
    of the feature representations minus the identity matrix.
    
    Parameters:
    - codes (np.ndarray): A 2D array of shape (batch_size, feature_dim), 
                             representing a batch of feature vectors.
    - corr_type (str): Either "correlation" for the Pearson correlation or "spearman" 
                       for the Spearman rank correlation.
    
    Returns:
    - float: The Frobenius norm of the (correlation matrix - identity matrix).
    """
    # Compute the correlation or Spearman correlation matrix
    if corr_type == "correlation":
        corr_matrix = np.corrcoef(codes, rowvar=False)
    elif corr_type == "spearman":
        corr_matrix, _ = spearmanr(codes, axis=0)
    else:
        raise ValueError("corr_type must be 'correlation' or 'spearman'")

    # Subtract the identity matrix
    identity_matrix = np.eye(corr_matrix.shape[0])
    diff_matrix = corr_matrix - identity_matrix

    # Compute and return the Frobenius norm
    frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
    return frobenius_norm
