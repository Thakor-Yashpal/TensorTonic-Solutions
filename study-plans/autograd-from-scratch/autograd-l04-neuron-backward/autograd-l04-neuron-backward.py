import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    preactivation = torch.dot(weights, inputs) + bias
    output = torch.tanh(preactivation)

    delta = upstream_gradient * (1 - output ** 2)


    grad_inputs = delta * weights
    grad_weights = delta * inputs
    grad_bias = delta

    return output, grad_inputs, grad_weights, grad_bias