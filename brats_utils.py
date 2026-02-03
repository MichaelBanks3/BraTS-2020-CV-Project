"""
FAST BraTS Dataset - Uses preprocessed .npy files

This is 10-50x faster than loading from .nii files because:
1. .npy files are already extracted slices (no 3D volume loading)
2. .npy files are uncompressed (no decompression overhead)
3. .npy files are memory-mapped friendly

Usage:
    # First run preprocessing (once):
    python preprocess_to_npy.py --data_root /path/to/raw --output_dir /path/to/preprocessed
    
    # Then use this dataset:
    from brats_utils_fast import BraTSFastDataset
    train_dataset = BraTSFastDataset('/path/to/preprocessed', split='train', augment=True)
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path
import numpy as np
import json
from scipy.ndimage import rotate


class BraTSFastDataset(Dataset):
    """
    Fast BraTS dataset that loads preprocessed .npy files
    
    Expected speedup: 10-50x compared to loading .nii files
    """
    
    def __init__(self, preprocessed_dir: str, split: str = 'train', augment: bool = False):
        """
        Args:
            preprocessed_dir: Path to directory with preprocessed .npy files
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.augment = augment
        self.split = split
        
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
        
        # Apply augmentation if enabled
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Z-Score normalize each channel independently
        for c in range(img.shape[0]):
            channel = img[c]
            if np.std(channel) > 0:
                img[c] = (channel - np.mean(channel)) / np.std(channel)
            else:
                img[c] = 0
        
        # Ensure mask values are clean (0, 1, 2, 3)
        mask = np.clip(np.round(mask), 0, 3).astype(np.int64)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).long()
        
        return img_tensor, mask_tensor
    
    def _augment(self, img, mask):
        """
        Apply data augmentation
        
        Args:
            img: (C, H, W) numpy array
            mask: (H, W) numpy array
        """
        # Random rotation between -15 and +15 degrees
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            # Rotate each channel
            for c in range(img.shape[0]):
                img[c] = rotate(img[c], angle, reshape=False, order=1, mode='nearest')
            mask = rotate(mask, angle, reshape=False, order=0, mode='nearest')
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()  # Flip along W dimension
            mask = np.flip(mask, axis=1).copy()
        
        return img, mask
    
    def get_sample_weights(self, tumor_oversample_factor: float = 5.0):
        """
        Get weights for WeightedRandomSampler to oversample tumor slices
        
        Usage:
            weights = dataset.get_sample_weights(tumor_oversample_factor=5.0)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
        """
        weights = []
        for slice_info in self.slices:
            if slice_info['has_tumor']:
                weights.append(tumor_oversample_factor)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float)
    
    def get_class_pixel_counts(self, max_samples: int = 1000):
        """
        Count pixels per class (for computing class weights)
        
        Args:
            max_samples: Number of samples to count (for speed)
        """
        counts = np.zeros(4, dtype=np.int64)
        
        indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)
        
        for idx in indices:
            slice_info = self.slices[idx]
            mask = np.load(self.preprocessed_dir / 'masks' / f"{slice_info['name']}.npy")
            for c in range(4):
                counts[c] += np.sum(mask == c)
        
        return counts


def create_weighted_sampler(dataset: BraTSFastDataset, tumor_oversample_factor: float = 5.0):
    """
    Convenience function to create a WeightedRandomSampler
    """
    weights = dataset.get_sample_weights(tumor_oversample_factor)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def compute_class_weights(dataset: BraTSFastDataset, max_samples: int = 1000):
    """
    Compute class weights inversely proportional to pixel frequency
    
    Returns weights suitable for nn.CrossEntropyLoss(weight=...)
    """
    counts = dataset.get_class_pixel_counts(max_samples)
    
    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (4 * counts + 1)  # +1 to avoid division by zero
    
    # Normalize so mean weight is 1
    weights = weights / weights.mean()
    
    # Cap background weight (it's huge otherwise)
    weights[0] = min(weights[0], 0.5)
    
    print(f"Computed class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# LEGACY DATASET (for comparison / if you don't want to preprocess)
# ============================================================================

class BraTSMultiClassDataset(Dataset):
    """
    Original slow dataset that loads from .nii files
    
    DEPRECATED: Use BraTSFastDataset instead for 10-50x speedup
    """
    def __init__(self, root_dir, start, stop, augment=False):
        import nibabel as nib
        self.nib = nib
        
        self.root_dir = Path(root_dir)
        self.patient_folders = sorted(list(self.root_dir.glob("BraTS20_Training_*")))
        self.valid_slices = []
        self.augment = augment
        self.start = start
        
        print("WARNING: Using slow dataset. Consider using BraTSFastDataset instead.")
        print("Preprocessing: Scanning for ALL slices with brain tissue...")
        
        patients_to_process = self.patient_folders[start:stop]
        tumor_count = 0
        
        for p_idx, path in enumerate(patients_to_process): 
            patient_id = path.name
            mask_path = path / f"{patient_id}_seg.nii"
            img_path = path / f"{patient_id}_t1ce.nii"
            
            try:
                mask_3d = nib.load(mask_path).get_fdata()
                img_3d = nib.load(img_path).get_fdata()
            except Exception as e:
                print(f"Warning: Could not load data for {patient_id}: {e}")
                continue
            
            for i in range(mask_3d.shape[2]):
                img_slice = img_3d[:, :, i]
                
                if np.std(img_slice) > 0.1 and np.count_nonzero(img_slice) > 1000:
                    has_tumor = np.sum(mask_3d[:, :, i]) > 0
                    self.valid_slices.append((p_idx, i, has_tumor))
                    if has_tumor:
                        tumor_count += 1
                    
        print(f"Found {len(self.valid_slices)} valid slices")
        print(f"  → {tumor_count} with tumor, {len(self.valid_slices) - tumor_count} without")

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, index):
        patient_idx, slice_idx, _ = self.valid_slices[index]
        patient_path = self.patient_folders[self.start + patient_idx]
        patient_id = patient_path.name
        
        img_path = patient_path / f"{patient_id}_t1ce.nii"
        mask_path = patient_path / f"{patient_id}_seg.nii"
        
        # SLOW: Loading entire 3D volume just for one slice
        img_slice = self.nib.load(img_path).get_fdata()[:, :, slice_idx]
        mask_slice = self.nib.load(mask_path).get_fdata()[:, :, slice_idx]
        
        if self.augment:
            img_slice, mask_slice = self._augment(img_slice, mask_slice)
        
        if np.std(img_slice) > 0:
            img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)
        else:
            img_slice = img_slice * 0
        
        mask_slice[mask_slice == 4] = 3
        mask_slice = np.clip(np.round(mask_slice), 0, 3).astype(np.int64)
        
        img_tensor = torch.from_numpy(img_slice.copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice.copy()).long()
        
        return img_tensor, mask_tensor
    
    def _augment(self, img, mask):
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = rotate(img, angle, reshape=False, order=1, mode='nearest')
            mask = rotate(mask, angle, reshape=False, order=0, mode='nearest')
        
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        return img, mask
    
    def get_sample_weights(self, tumor_oversample_factor=5.0):
        weights = []
        for _, _, has_tumor in self.valid_slices:
            weights.append(tumor_oversample_factor if has_tumor else 1.0)
        return torch.tensor(weights, dtype=torch.float)
