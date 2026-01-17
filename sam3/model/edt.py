# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Triton kernel for euclidean distance transform (EDT) - PATCHED for CPU/MPS"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    # assert data.dim() == 3
    # assert data.is_cuda # Removed constraint
    
    device = data.device
    dtype = data.dtype
    
    # Move to CPU numpy
    # Ensure it's boolean/binary for scipy
    data_np = data.detach().cpu().numpy()
    
    B, H, W = data_np.shape
    # Output should be float
    output_np = np.empty((B, H, W), dtype=np.float32)
    
    for b in range(B):
        # scipy.ndimage.distance_transform_edt computes distance to the nearest ZERO (background).
        # Assuming data_np[b] is 1 for foreground, 0 for background.
        output_np[b] = distance_transform_edt(data_np[b])
        
    return torch.from_numpy(output_np).to(device)
