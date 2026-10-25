import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    x_arr = np.array(x)
    
    # Apply the sigmoid formula: 1 / (1 + e^-x)
    result = 1 / (1 + np.exp(-x_arr))
    
    if isinstance(x, (int, float)):
        return float(result)
        
    return result