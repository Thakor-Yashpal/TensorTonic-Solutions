import torch
import triton
import triton.language as tl

@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the 1D program ID and compute block offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a tail mask to prevent out-of-bounds memory access
    tail_mask = offsets < n
    
    # Load the input elements and their corresponding 0/1 mask values
    x = tl.load(x_ptr + offsets, mask=tail_mask)
    m = tl.load(mask_ptr + offsets, mask=tail_mask)
    
    # Compute the inverted scaling factor once per block
    scale = 1.0 / (1.0 - p)
    
    # Apply the elementwise inverted dropout formula
    y = x * m * scale
    
    # Store back to global memory
    tl.store(out_ptr + offsets, y, mask=tail_mask)

def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )