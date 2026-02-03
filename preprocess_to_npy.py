"""
PREPROCESSING SCRIPT - Run this ONCE before training

This script:
1. Scans all patient folders
2. Extracts valid 2D slices (with brain tissue)
3. Saves each slice as a small .npy file
4. Creates a metadata JSON for fast dataset loading

Expected speedup: 10-50x faster training

Usage:
    python preprocess_to_npy.py --data_root /path/to/MICCAI_BraTS2020_TrainingData --output_dir /path/to/preprocessed
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import time


def preprocess_brats_dataset(data_root: Path, output_dir: Path, modalities: list = None):
    """
    Extract all valid 2D slices from BraTS .nii files and save as .npy
    
    Args:
        data_root: Path to MICCAI_BraTS2020_TrainingData
        output_dir: Where to save preprocessed .npy files
        modalities: List of modalities to extract (default: just t1ce)
    """
    if modalities is None:
        modalities = ['t1ce']  # Add 't1', 't2', 'flair' for multi-modality later
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    
    # Find all patient folders
    patient_folders = sorted(list(data_root.glob("BraTS20_Training_*")))
    print(f"Found {len(patient_folders)} patients")
    
    # Metadata to track all slices
    metadata = {
        'slices': [],  # List of {'patient_id': str, 'slice_idx': int, 'has_tumor': bool}
        'modalities': modalities,
        'num_patients': len(patient_folders),
    }
    
    total_slices = 0
    tumor_slices = 0
    start_time = time.time()
    
    for patient_path in tqdm(patient_folders, desc="Processing patients"):
        patient_id = patient_path.name
        
        # Load mask first to check which slices have brain/tumor
        mask_path = patient_path / f"{patient_id}_seg.nii"
        if not mask_path.exists():
            mask_path = patient_path / f"{patient_id}_seg.nii.gz"
        
        try:
            mask_3d = nib.load(str(mask_path)).get_fdata()
        except Exception as e:
            print(f"Warning: Could not load mask for {patient_id}: {e}")
            continue
        
        # Load image(s)
        images_3d = {}
        for modality in modalities:
            img_path = patient_path / f"{patient_id}_{modality}.nii"
            if not img_path.exists():
                img_path = patient_path / f"{patient_id}_{modality}.nii.gz"
            
            try:
                images_3d[modality] = nib.load(str(img_path)).get_fdata()
            except Exception as e:
                print(f"Warning: Could not load {modality} for {patient_id}: {e}")
                break
        
        if len(images_3d) != len(modalities):
            continue  # Skip patient if any modality failed to load
        
        # Process each slice
        num_slices = mask_3d.shape[2]
        
        for slice_idx in range(num_slices):
            # Get the image slice (use first modality to check validity)
            img_slice = images_3d[modalities[0]][:, :, slice_idx]
            
            # Check if slice has brain tissue (same criteria as before)
            if np.std(img_slice) > 0.1 and np.count_nonzero(img_slice) > 1000:
                # This is a valid slice - save it
                
                # Extract mask slice
                mask_slice = mask_3d[:, :, slice_idx].astype(np.int8)
                
                # Remap labels: 0, 1, 2, 4 → 0, 1, 2, 3
                mask_slice[mask_slice == 4] = 3
                
                has_tumor = np.any(mask_slice > 0)
                
                # Create unique filename
                slice_name = f"{patient_id}_slice{slice_idx:03d}"
                
                # Stack all modalities into single array (C, H, W)
                if len(modalities) == 1:
                    img_stack = img_slice.astype(np.float32)
                else:
                    img_stack = np.stack([images_3d[m][:, :, slice_idx] for m in modalities], axis=0).astype(np.float32)
                
                # Save to .npy files
                np.save(output_dir / 'images' / f"{slice_name}.npy", img_stack)
                np.save(output_dir / 'masks' / f"{slice_name}.npy", mask_slice)
                
                # Track metadata
                metadata['slices'].append({
                    'name': slice_name,
                    'patient_id': patient_id,
                    'slice_idx': slice_idx,
                    'has_tumor': bool(has_tumor)
                })
                
                total_slices += 1
                if has_tumor:
                    tumor_slices += 1
    
    # Save metadata
    metadata['total_slices'] = total_slices
    metadata['tumor_slices'] = tumor_slices
    metadata['non_tumor_slices'] = total_slices - tumor_slices
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total slices extracted: {total_slices:,}")
    print(f"  - With tumor: {tumor_slices:,} ({tumor_slices/total_slices*100:.1f}%)")
    print(f"  - Without tumor: {total_slices - tumor_slices:,} ({(total_slices-tumor_slices)/total_slices*100:.1f}%)")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {output_dir / 'images'} ({total_slices} .npy files)")
    print(f"  - {output_dir / 'masks'} ({total_slices} .npy files)")
    print(f"  - {output_dir / 'metadata.json'}")
    
    return metadata


def create_train_val_test_split(output_dir: Path, train_patients: tuple, val_patients: tuple, test_patients: tuple):
    """
    Create separate metadata files for train/val/test splits based on patient indices
    
    Args:
        output_dir: Path to preprocessed data
        train_patients: (start, stop) patient indices for training
        val_patients: (start, stop) patient indices for validation  
        test_patients: (start, stop) patient indices for test
    """
    output_dir = Path(output_dir)
    
    # Load full metadata
    with open(output_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Get sorted unique patient IDs to determine split
    all_patient_ids = sorted(list(set(s['patient_id'] for s in metadata['slices'])))
    
    train_ids = set(all_patient_ids[train_patients[0]:train_patients[1]])
    val_ids = set(all_patient_ids[val_patients[0]:val_patients[1]])
    test_ids = set(all_patient_ids[test_patients[0]:test_patients[1]])
    
    # Split slices
    splits = {
        'train': {'slices': [], 'patient_ids': list(train_ids)},
        'val': {'slices': [], 'patient_ids': list(val_ids)},
        'test': {'slices': [], 'patient_ids': list(test_ids)}
    }
    
    for slice_info in metadata['slices']:
        pid = slice_info['patient_id']
        if pid in train_ids:
            splits['train']['slices'].append(slice_info)
        elif pid in val_ids:
            splits['val']['slices'].append(slice_info)
        elif pid in test_ids:
            splits['test']['slices'].append(slice_info)
    
    # Save split files
    for split_name, split_data in splits.items():
        split_data['modalities'] = metadata['modalities']
        split_data['total_slices'] = len(split_data['slices'])
        split_data['tumor_slices'] = sum(1 for s in split_data['slices'] if s['has_tumor'])
        
        with open(output_dir / f'{split_name}_metadata.json', 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"{split_name}: {split_data['total_slices']} slices from {len(split_data['patient_ids'])} patients "
              f"({split_data['tumor_slices']} with tumor)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess BraTS data to .npy files')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to MICCAI_BraTS2020_TrainingData')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save preprocessed files')
    parser.add_argument('--modalities', type=str, nargs='+', default=['t1ce'],
                        help='Modalities to extract (default: t1ce). Options: t1, t1ce, t2, flair')
    
    args = parser.parse_args()
    
    # Run preprocessing
    metadata = preprocess_brats_dataset(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        modalities=args.modalities
    )
    
    # Create train/val/test splits (same as your original: 0-200, 200-300, 300-369)
    print("\nCreating train/val/test splits...")
    create_train_val_test_split(
        output_dir=Path(args.output_dir),
        train_patients=(0, 200),
        val_patients=(200, 300),
        test_patients=(300, 369)
    )
    
    print("\n✓ Done! You can now use brats_utils_fast.py for training.")
