import torch
import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    arr = torch.tensor(matrix, dtype=torch.float64)
    
    # Map the requested norm to PyTorch's p-norm order parameter
    p_order = {'l1': 1, 'l2': 2, 'max': float('inf')}[norm_type]
    
    keepdim = axis is not None
    norms = torch.linalg.vector_norm(arr, ord=p_order, dim=axis, keepdim=keepdim)
    
    norms = torch.where(norms == 0.0, torch.tensor(1.0, dtype=norms.dtype), norms)
    
    return (arr / norms).numpy()