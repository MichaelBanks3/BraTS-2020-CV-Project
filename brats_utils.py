import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate

class BraTSMultiClassDataset(Dataset):
    """
    ** NEW FOR MODEL 4 **
    
    Multi-class version of BraTS dataset with the following improvements:
    1. Predicts all 4 tumor labels (NCR/NET, ED, ET) instead of binary
    2. Includes slices WITHOUT tumors (to teach model when there's no tumor)
    3. Supports data augmentation (rotation, horizontal flip)
    4. Fixes the indexing bug from Model 3
    
    Label mapping:
    - Original BraTS labels: 0 (background), 1 (NCR/NET), 2 (ED), 4 (ET)
    - Model output indices: 0 (background), 1 (NCR/NET), 2 (ED), 3 (ET)
    """
    def __init__(self, root_dir, start, stop, augment=False):
        self.root_dir = Path(root_dir)
        self.patient_folders = sorted(list(self.root_dir.glob("BraTS20_Training_*")))
        self.valid_slices = []
        self.augment = augment  # NEW: Enable/disable augmentation
        self.start = start  # NEW: Store start index to fix bug
        
        print("Preprocessing: Scanning for ALL slices with brain tissue...")
        
        # Only process a subset for quick testing/development
        patients_to_process = self.patient_folders[start:stop]
        
        for p_idx, path in enumerate(patients_to_process): 
            patient_id = path.name
            mask_path = path / f"{patient_id}_seg.nii"
            img_path = path / f"{patient_id}_t1ce.nii"
            
            # Load 3D mask and image
            try:
                mask_3d = nib.load(mask_path).get_fdata()
                img_3d = nib.load(img_path).get_fdata()
            except nib.filebased.FileBasedError:
                print(f"Warning: Could not load data for {patient_id}")
                continue
            
            # NEW FOR MODEL 4: Include ALL slices with brain tissue, not just tumor slices
            for i in range(mask_3d.shape[2]):
                img_slice = img_3d[:, :, i]
                
                # Check if slice has brain tissue (not a blank/empty slice)
                # We detect blank slices by checking if the slice has very low standard deviation
                # or is mostly zeros. Real brain tissue has texture and variation.
                if np.std(img_slice) > 0.1 and np.count_nonzero(img_slice) > 1000:
                    # This slice has brain tissue, so include it
                    # (It might have tumor or might not - we want both!)
                    self.valid_slices.append((p_idx, i))
                    
        print(f"Success! Found {len(self.valid_slices)} valid slices with brain tissue.")
        
        # Count how many have tumors vs. don't have tumors
        tumor_count = 0
        for p_idx, slice_idx in self.valid_slices:
            patient_path = self.patient_folders[start + p_idx]
            patient_id = patient_path.name
            mask_path = patient_path / f"{patient_id}_seg.nii"
            mask_3d = nib.load(mask_path).get_fdata()
            if np.sum(mask_3d[:, :, slice_idx]) > 0:
                tumor_count += 1
        
        print(f"  → {tumor_count} slices with tumor ({tumor_count/len(self.valid_slices)*100:.1f}%)")
        print(f"  → {len(self.valid_slices) - tumor_count} slices without tumor ({(len(self.valid_slices)-tumor_count)/len(self.valid_slices)*100:.1f}%)")

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, index):
        patient_idx, slice_idx = self.valid_slices[index]
        
        # BUG FIX: Add self.start to correctly index into full patient_folders list
        patient_path = self.patient_folders[self.start + patient_idx]
        patient_id = patient_path.name
        
        img_path = patient_path / f"{patient_id}_t1ce.nii"
        mask_path = patient_path / f"{patient_id}_seg.nii"
        
        # Load and extract slice
        img_slice = nib.load(img_path).get_fdata()[:, :, slice_idx]
        mask_slice = nib.load(mask_path).get_fdata()[:, :, slice_idx]
        
        # NEW FOR MODEL 4: Apply data augmentation (only during training)
        if self.augment:
            img_slice, mask_slice = self._augment(img_slice, mask_slice)
        
        # Z-Score Normalization (same as before)
        if np.std(img_slice) > 0:
            img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)
        else:
            img_slice = img_slice * 0
        
        # NEW FOR MODEL 4: Keep multi-class labels instead of binarizing
        # BraTS labels are: 0 (background), 1 (NCR/NET), 2 (ED), 4 (ET)
        # We need to remap label 4 → 3 for PyTorch (expects 0, 1, 2, 3)
        mask_slice[mask_slice == 4] = 3  # Remap ET from 4 to 3
        # Now labels are: 0 (background), 1 (NCR/NET), 2 (ED), 3 (ET)
        
        # Convert to Tensor
        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0)  # Shape: (1, 240, 240)
        
        # NEW FOR MODEL 4: Mask is now a long tensor (class indices, not float probabilities)
        mask_tensor = torch.from_numpy(mask_slice).long().squeeze()  # Shape: (240, 240)
        
        return img_tensor, mask_tensor
    
    def _augment(self, img, mask):
        """
        NEW FOR MODEL 4: Apply data augmentation
        
        Augmentations applied:
        1. Random rotation (±15 degrees) - safe for medical images
        2. Random horizontal flip - brain is roughly symmetric
        
        We apply the SAME transformation to both image and mask
        so they stay aligned.
        """
        # Random rotation between -15 and +15 degrees
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = rotate(img, angle, reshape=False, order=1)  # order=1 = bilinear
            mask = rotate(mask, angle, reshape=False, order=0)  # order=0 = nearest neighbor
        
        # Random horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        return img, mask
