import torch
import torch.nn.functional as F

def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:
    B, C, H, W = images.shape
    N_H = H // patch_height
    N_W = W // patch_width
    
    tokens = F.unfold(
        images, 
        kernel_size=(patch_height, patch_width), 
        stride=(patch_height, patch_width)
    )
    
    # Transpose to get the expected sequence shape: (B, patch_count, patch_features)
    tokens = tokens.transpose(1, 2)
    
    # Generate coordinates using modern meshgrid with explicit indexing and flatten
    r = torch.arange(N_H, dtype=torch.int64, device=images.device)
    c = torch.arange(N_W, dtype=torch.int64, device=images.device)
    
    grid_r, grid_c = torch.meshgrid(r, c, indexing='ij')
    coordinates = torch.stack((grid_r.flatten(), grid_c.flatten()), dim=1)
    
    return {
        "tokens": tokens,
        "coordinates": coordinates
    }