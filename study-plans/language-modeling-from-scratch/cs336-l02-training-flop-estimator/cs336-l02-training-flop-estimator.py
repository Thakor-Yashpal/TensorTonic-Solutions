def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict[str, int]:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    Each matmul entry is a list/tuple of [B, D, K].
    """
    # Sum up matrix multiplication FLOPs (2 * B * D * K per matmul)
    matmul_flops = sum(2 * b * d * k for b, d, k in matmuls)
    
    # Compute forward, backward, and total FLOPs
    forward_flops = matmul_flops + attention_flops
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_flops": total_flops,
    }