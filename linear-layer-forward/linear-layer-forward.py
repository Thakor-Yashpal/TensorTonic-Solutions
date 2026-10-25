import torch
import torch.nn.functional as F

def linear_layer_forward(X: list, W: list, b: list) -> list:
    # Convert inputs to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    W_tensor = torch.tensor(W, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    
    # Compute X @ W.T + b using PyTorch's optimized linear function
    Y_tensor = F.linear(X_tensor, W_tensor.T, b_tensor)
    
    # Convert the resulting tensor back to a nested Python list
    return Y_tensor.tolist()