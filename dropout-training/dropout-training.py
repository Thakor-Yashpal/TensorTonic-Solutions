import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.array(x, dtype=float)
    
    if rng is not None:
        rand_vals = rng.random(x_arr.shape)
    else:
        rand_vals = np.random.random(x_arr.shape)
        
    keep_mask = (rand_vals >= p).astype(float)

    dropout_pattern = keep_mask / (1.0 - p)
    
    # Apply the mask to the input array
    output = x_arr * dropout_pattern
    
    return output, dropout_pattern