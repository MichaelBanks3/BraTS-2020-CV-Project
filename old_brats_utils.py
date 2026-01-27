import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import numpy as np

class BraTSSmartDataset(Dataset):
    """
    Loads and preprocesses BraTS MRI data, dynamically filtering for slices 
    that contain a tumor annotation.
    """
    def __init__(self, root_dir, start, stop):
        self.root_dir = Path(root_dir)
        self.patient_folders = sorted(list(self.root_dir.glob("BraTS20_Training_*")))
        self.valid_slices = [] 
        
        print("Preprocessing: Scanning for slices with tumors...")
        
        # Only process a subset for quick testing/development
        patients_to_process = self.patient_folders[start:stop]
        
        for p_idx, path in enumerate(patients_to_process): 
            patient_id = path.name
            mask_path = path / f"{patient_id}_seg.nii"
            
            # Load 3D mask
            try:
                mask_3d = nib.load(mask_path).get_fdata()
            except nib.filebased.FileBasedError:
                print(f"Warning: Could not load mask for {patient_id}")
                continue
            
            # Iterate through slices
            for i in range(mask_3d.shape[2]):
                # Tumor check: only keep slice if it has > 50 tumor pixels
                if np.sum(mask_3d[:, :, i]) > 50: 
                    self.valid_slices.append((p_idx, i))
                    
        print(f"Success! Found {len(self.valid_slices)} valid slices containing tumors.")

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, index):
        patient_idx, slice_idx = self.valid_slices[index]
        patient_path = self.patient_folders[patient_idx]
        patient_id = patient_path.name
        
        img_path = patient_path / f"{patient_id}_t1ce.nii"
        mask_path = patient_path / f"{patient_id}_seg.nii"
        
        # Load and extract slice
        img_slice = nib.load(img_path).get_fdata()[:, :, slice_idx]
        mask_slice = nib.load(mask_path).get_fdata()[:, :, slice_idx]
        
        # Z-Score Normalization
        if np.std(img_slice) > 0:
            img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)
        else:
            img_slice = img_slice * 0
            
        # Binarize mask
        mask_slice[mask_slice > 0] = 1.0
        
        # Convert to Tensor (add channel dimension)
        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)
        
        return img_tensor, mask_tensor