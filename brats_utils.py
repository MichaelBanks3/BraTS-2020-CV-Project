"""
FAST BraTS Dataset - Model 5 Optimized

Key optimization: Augmentation moved to GPU (in training loop)
- Dataset only does: load .npy + normalize (fast)
- Augmentation done on GPU after .to(device) (essentially free)

This fixes the 7 it/s → 2 it/s slowdown caused by CPU-bound scipy.rotate
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path
import numpy as np
import json


class BraTSFastDataset(Dataset):
    """
    Fast BraTS dataset - NO CPU AUGMENTATION
    
    Augmentation is done on GPU via gpu_augment() function
    """
    
    def __init__(self, preprocessed_dir: str, split: str = 'train', augment: bool = False):
        """
        Args:
            preprocessed_dir: Path to directory with preprocessed .npy files
            split: 'train', 'val', or 'test'
            augment: IGNORED - augmentation now done on GPU in training loop
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        
        # Note: augment parameter is kept for API compatibility but ignored
        # Augmentation should be done on GPU using gpu_augment() function
        if augment:
            print("  Note: CPU augmentation disabled. Use gpu_augment() in training loop instead.")
        
        # Load metadata for this split
        metadata_path = self.preprocessed_dir / f'{split}_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Did you run preprocess_to_npy.py first?"
            )
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.slices = self.metadata['slices']
        self.num_modalities = len(self.metadata['modalities'])
        
        # Print summary
        tumor_count = sum(1 for s in self.slices if s['has_tumor'])
        print(f"Loaded {split} dataset: {len(self.slices)} slices")
        print(f"  → {tumor_count} with tumor ({tumor_count/len(self.slices)*100:.1f}%)")
        print(f"  → {len(self.slices) - tumor_count} without tumor")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, index):
        slice_info = self.slices[index]
        slice_name = slice_info['name']
        
        # Load preprocessed .npy files (FAST!)
        img = np.load(self.preprocessed_dir / 'images' / f'{slice_name}.npy')
        mask = np.load(self.preprocessed_dir / 'masks' / f'{slice_name}.npy')
        
        # Handle single vs multi-modality
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # Add channel dim: (H, W) -> (1, H, W)
        
        # Vectorized Z-Score normalization (FAST - no Python loops)
        # Compute mean and std for all channels at once
        img = img.astype(np.float32)
        
        # Per-channel normalization using numpy broadcasting
        # Shape: (C, H, W) -> compute stats over (H, W) for each channel
        means = img.mean(axis=(1, 2), keepdims=True)  # (C, 1, 1)
        stds = img.std(axis=(1, 2), keepdims=True)    # (C, 1, 1)
        stds = np.maximum(stds, 1e-8)  # Avoid division by zero
        img = (img - means) / stds
        
        # Ensure mask values are clean (0, 1, 2, 3)
        mask = np.clip(np.round(mask), 0, 3).astype(np.int64)
        
        # Convert to tensors - use np.ascontiguousarray to avoid extra copy
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        
        return img_tensor, mask_tensor
    
    def get_sample_weights(self, tumor_oversample_factor: float = 5.0):
        """Get weights for WeightedRandomSampler to oversample tumor slices"""
        weights = []
        for slice_info in self.slices:
            if slice_info['has_tumor']:
                weights.append(tumor_oversample_factor)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float)
    
    def get_class_pixel_counts(self, max_samples: int = 1000):
        """Count pixels per class (for computing class weights)"""
        counts = np.zeros(4, dtype=np.int64)
        indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)
        
        for idx in indices:
            slice_info = self.slices[idx]
            mask = np.load(self.preprocessed_dir / 'masks' / f"{slice_info['name']}.npy")
            for c in range(4):
                counts[c] += np.sum(mask == c)
        
        return counts


# =============================================================================
# GPU AUGMENTATION - Call this in training loop AFTER moving to device
# =============================================================================

def gpu_augment(images, masks, p_flip=0.5, p_rotate=0.5, max_angle=15):
    """
    Apply augmentation on GPU (FAST!)
    
    Call this in training loop:
        images, masks = images.to(device), masks.to(device)
        if training:
            images, masks = gpu_augment(images, masks)
    
    Args:
        images: (B, C, H, W) tensor on GPU
        masks: (B, H, W) tensor on GPU  
        p_flip: probability of horizontal flip
        p_rotate: probability of rotation
        max_angle: max rotation angle in degrees
    
    Returns:
        augmented images and masks (same shapes)
    """
    B = images.shape[0]
    device = images.device
    
    # Random horizontal flip (per-sample)
    if p_flip > 0:
        flip_mask = torch.rand(B, device=device) < p_flip
        if flip_mask.any():
            # Flip images: (B, C, H, W) -> flip along W (dim=3)
            images[flip_mask] = torch.flip(images[flip_mask], dims=[3])
            # Flip masks: (B, H, W) -> flip along W (dim=2)
            masks[flip_mask] = torch.flip(masks[flip_mask], dims=[2])
    
    # Random rotation (per-sample)
    if p_rotate > 0:
        rotate_mask = torch.rand(B, device=device) < p_rotate
        if rotate_mask.any():
            # Generate random angles for samples that need rotation
            angles = torch.zeros(B, device=device)
            angles[rotate_mask] = torch.empty(rotate_mask.sum(), device=device).uniform_(-max_angle, max_angle)
            
            # Only rotate samples that need it
            indices = torch.where(rotate_mask)[0]
            for idx in indices:
                angle = angles[idx].item()
                images[idx] = _rotate_tensor(images[idx], angle)
                masks[idx] = _rotate_mask(masks[idx], angle)
    
    return images, masks


def _rotate_tensor(img, angle_deg):
    """
    Rotate a single image tensor on GPU
    
    Args:
        img: (C, H, W) tensor
        angle_deg: rotation angle in degrees
    
    Returns:
        rotated (C, H, W) tensor
    """
    angle_rad = angle_deg * np.pi / 180
    
    # Create rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Affine matrix for rotation around center
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=img.dtype, device=img.device).unsqueeze(0)  # (1, 2, 3)
    
    # Add batch dimension for grid_sample
    img_4d = img.unsqueeze(0)  # (1, C, H, W)
    
    # Create sampling grid
    grid = F.affine_grid(theta, img_4d.size(), align_corners=False)
    
    # Sample with bilinear interpolation
    rotated = F.grid_sample(img_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    return rotated.squeeze(0)  # Back to (C, H, W)


def _rotate_mask(mask, angle_deg):
    """
    Rotate a single mask tensor on GPU (nearest neighbor interpolation)
    
    Args:
        mask: (H, W) tensor (long dtype)
        angle_deg: rotation angle in degrees
    
    Returns:
        rotated (H, W) tensor
    """
    angle_rad = angle_deg * np.pi / 180
    
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=torch.float32, device=mask.device).unsqueeze(0)
    
    # Mask needs to be float for grid_sample, then convert back
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    
    grid = F.affine_grid(theta, mask_4d.size(), align_corners=False)
    
    # Use nearest neighbor for masks to preserve integer labels
    rotated = F.grid_sample(mask_4d, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    
    return rotated.squeeze(0).squeeze(0).long()  # Back to (H, W)


# =============================================================================
# FAST GPU AUGMENTATION (BATCHED) - Even faster for large batches
# =============================================================================

def gpu_augment_fast(images, masks, p_flip=0.5):
    """
    Ultra-fast augmentation - flip only (rotation is slower even on GPU)
    
    For maximum throughput, use this instead of gpu_augment()
    Rotation adds ~10-20% overhead; flip is essentially free
    
    Args:
        images: (B, C, H, W) tensor on GPU
        masks: (B, H, W) tensor on GPU
        p_flip: probability of horizontal flip
    
    Returns:
        augmented images and masks
    """
    B = images.shape[0]
    device = images.device
    
    # Random horizontal flip
    flip_mask = torch.rand(B, device=device) < p_flip
    if flip_mask.any():
        images[flip_mask] = torch.flip(images[flip_mask], dims=[3])
        masks[flip_mask] = torch.flip(masks[flip_mask], dims=[2])
    
    return images, masks


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_weighted_sampler(dataset: BraTSFastDataset, tumor_oversample_factor: float = 5.0):
    """Convenience function to create a WeightedRandomSampler"""
    weights = dataset.get_sample_weights(tumor_oversample_factor)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def compute_class_weights(dataset: BraTSFastDataset, max_samples: int = 1000):
    """Compute class weights inversely proportional to pixel frequency"""
    counts = dataset.get_class_pixel_counts(max_samples)
    
    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (4 * counts + 1)
    
    # Normalize so mean weight is 1
    weights = weights / weights.mean()
    
    # Cap background weight
    weights[0] = min(weights[0], 0.5)
    
    print(f"Computed class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)
