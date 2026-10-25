import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    # Clone parameters to avoid modifying inputs in-place
    W = [w.clone() for w in weights]
    b = [bias.clone() for bias in biases]
    loss_history = []
    
    for _ in range(steps):
        # 1. Forward Pass
        H = inputs
        activations = [H]
        
        for w_i, b_i in zip(W, b):
            Z = torch.matmul(H, w_i.T) + b_i
            H = torch.tanh(Z)
            activations.append(H)
            
        preds = H.reshape(-1)
        
        # Calculate SSE loss
        loss = torch.sum((preds - targets) ** 2)
        loss_history.append(loss)
        
        # 2. Backward Pass (Manual Backpropagation)
        # Derivative of SSE loss with respect to predictions
        grad_preds = 2.0 * (preds - targets)
        
        # delta for final layer = grad_loss * derivative of tanh(x)
        # derivative of tanh(x) is (1 - tanh(x)^2)
        delta = grad_preds.reshape(-1, 1) * (1.0 - activations[-1] ** 2)
        
        grad_W = []
        grad_b = []
        
        # Traverse layers in reverse to apply the chain rule
        for i in reversed(range(len(W))):
            H_prev = activations[i]
            
            # Gradient of weights: delta^T @ H_prev
            dW = torch.matmul(delta.T, H_prev)
            # Gradient of biases: sum of delta over the batch dimension
            db = torch.sum(delta, dim=0)
            
            grad_W.insert(0, dW)
            grad_b.insert(0, db)
            
            # Calculate delta for the previous layer (if not at the input layer yet)
            if i > 0:
                delta = torch.matmul(delta, W[i]) * (1.0 - H_prev ** 2)
                
        # 3. Parameter Update Step (SGD)
        for i in range(len(W)):
            W[i] = W[i] - learning_rate * grad_W[i]
            b[i] = b[i] - learning_rate * grad_b[i]
            
    # Final forward pass to compute outputs with the newly updated parameters
    H = inputs
    for w_i, b_i in zip(W, b):
        Z = torch.matmul(H, w_i.T) + b_i
        H = torch.tanh(Z)
        
    final_preds = H.reshape(-1)
    final_loss = torch.sum((final_preds - targets) ** 2)
    
    return final_preds, final_loss, W, b, loss_history