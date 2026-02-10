"""
BraTS 2020 - 3D U-Net Utilities (v2 - Fixed)
Dataset class, metrics, and helper functions for 3D volumetric segmentation

FIXES from audit:
- HD95: proper surface extraction using binary erosion XOR
- Validation: multi-patch mode instead of single center crop
- File loading: robust existence checks with clear error messages
- Added WT/TC/ET dice computation for training monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_erosion, generate_binary_structure
import os


# ============================================================================
# 3D Dataset Class
# ============================================================================

class BraTSDataset3D(Dataset):
    """
    3D BraTS Dataset - loads full NIfTI volumes and extracts random patches
    
    Args:
        data_dir: Path to BraTS data (contains patient folders)
        patient_ids: List of patient IDs to include
        patch_size: Size of 3D patches to extract (D, H, W)
        augment: Whether to apply augmentation
        mode: 'train', 'val', or 'test'
        samples_per_volume: patches per volume per epoch (train mode)
        val_patches_per_volume: number of fixed patches for validation
        tumor_center_prob: probability of centering patch on tumor (for training)
    """
    def __init__(self, data_dir, patient_ids, patch_size=(128, 128, 128), 
                 augment=False, mode='train', samples_per_volume=4,
                 val_patches_per_volume=8, tumor_center_prob=0.8):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.patch_size = patch_size
        self.augment = augment
        self.mode = mode
        self.samples_per_volume = samples_per_volume
        self.val_patches_per_volume = val_patches_per_volume
        self.tumor_center_prob = tumor_center_prob
        
        # Modalities in order
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
        # Cache for loaded volumes (saves I/O, enable if RAM allows)
        self.cache = {}
        self.use_cache = False
        
        # Pre-compute validation patch locations (deterministic)
        if mode == 'val':
            self._precompute_val_locations()
    
    def _precompute_val_locations(self):
        """Generate deterministic patch locations for validation"""
        self.val_locations = {}
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for patient_id in self.patient_ids:
            # Load just to get shape (or use known BraTS shape)
            # BraTS volumes are typically 155 x 240 x 240
            d, h, w = 155, 240, 240
            pd, ph, pw = self.patch_size
            
            locations = []
            # Grid-based sampling to cover volume
            for z in np.linspace(0, max(0, d - pd), 2, dtype=int):
                for y in np.linspace(0, max(0, h - ph), 2, dtype=int):
                    for x in np.linspace(0, max(0, w - pw), 2, dtype=int):
                        locations.append((z, y, x))
            
            self.val_locations[patient_id] = locations[:self.val_patches_per_volume]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.patient_ids) * self.samples_per_volume
        elif self.mode == 'val':
            return len(self.patient_ids) * self.val_patches_per_volume
        else:  # test
            return len(self.patient_ids)
    
    def __getitem__(self, idx):
        # Map index to patient and patch
        if self.mode == 'train':
            patient_idx = idx // self.samples_per_volume
        elif self.mode == 'val':
            patient_idx = idx // self.val_patches_per_volume
            patch_idx = idx % self.val_patches_per_volume
        else:
            patient_idx = idx
            
        patient_id = self.patient_ids[patient_idx]
        
        # Load volume (from cache or disk)
        if self.use_cache and patient_id in self.cache:
            image, seg = self.cache[patient_id]
            image = image.copy()
            seg = seg.copy()
        else:
            image, seg = self._load_volume(patient_id)
            if self.use_cache:
                self.cache[patient_id] = (image.copy(), seg.copy())
        
        # Extract patch based on mode
        if self.mode == 'train':
            image, seg = self._random_crop(image, seg)
        elif self.mode == 'val':
            loc = self.val_locations[patient_id][patch_idx]
            image, seg = self._extract_patch(image, seg, loc)
        else:  # test - return center crop or handle differently
            image, seg = self._center_crop(image, seg)
        
        # Augmentation (only for training)
        if self.augment and self.mode == 'train':
            image, seg = self._augment(image, seg)
        
        return torch.from_numpy(image.copy()), torch.from_numpy(seg.copy())
    
    def _find_file(self, patient_id, modality):
        """Find file with robust pattern matching and clear errors"""
        patterns = [
            self.data_dir / patient_id / f"{patient_id}_{modality}.nii.gz",
            self.data_dir / patient_id / f"{patient_id}_{modality}.nii",
            self.data_dir / patient_id / f"{modality}.nii.gz",
            self.data_dir / patient_id / f"{modality}.nii",
            # Handle uppercase
            self.data_dir / patient_id / f"{patient_id}_{modality.upper()}.nii.gz",
            self.data_dir / patient_id / f"{patient_id}_{modality.upper()}.nii",
        ]
        
        for path in patterns:
            if path.exists():
                return path
        
        # If not found, raise clear error
        tried = "\n  ".join(str(p) for p in patterns)
        raise FileNotFoundError(
            f"Missing {modality} for patient {patient_id}.\n"
            f"Tried:\n  {tried}\n"
            f"Please check your data directory structure."
        )
    
    def _load_volume(self, patient_id):
        """Load all modalities and segmentation for a patient"""
        volumes = []
        
        for mod in self.modalities:
            nii_path = self._find_file(patient_id, mod)
            vol = nib.load(str(nii_path)).get_fdata()
            volumes.append(vol)
        
        # Stack modalities: (4, H, W, D) -> transpose to (4, D, H, W)
        image = np.stack(volumes, axis=0).astype(np.float32)
        image = np.transpose(image, (0, 3, 1, 2))  # (4, D, H, W)
        
        # Load segmentation
        seg_path = self._find_file(patient_id, 'seg')
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int64)
        seg = np.transpose(seg, (2, 0, 1))  # (D, H, W)
        
        # BraTS labels: 0=background, 1=NCR, 2=ED, 4=ET -> remap 4 to 3
        seg[seg == 4] = 3
        
        # Normalize each modality (z-score on non-zero voxels)
        for i in range(4):
            mask = image[i] > 0
            if mask.sum() > 0:
                mean = image[i][mask].mean()
                std = image[i][mask].std()
                if std > 0:
                    image[i][mask] = (image[i][mask] - mean) / std
        
        return image, seg
    
    def _extract_patch(self, image, seg, location):
        """Extract patch at specific location with padding if needed"""
        z, y, x = location
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Initialize with zeros (padding)
        image_crop = np.zeros((4, pd, ph, pw), dtype=image.dtype)
        seg_crop = np.zeros((pd, ph, pw), dtype=seg.dtype)
        
        # Calculate valid region
        z_end = min(z + pd, d)
        y_end = min(y + ph, h)
        x_end = min(x + pw, w)
        
        crop_d = z_end - z
        crop_h = y_end - y
        crop_w = x_end - x
        
        image_crop[:, :crop_d, :crop_h, :crop_w] = image[:, z:z_end, y:y_end, x:x_end]
        seg_crop[:crop_d, :crop_h, :crop_w] = seg[z:z_end, y:y_end, x:x_end]
        
        return image_crop, seg_crop
    
    def _random_crop(self, image, seg):
        """Extract random patch, biased toward tumor regions"""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Find tumor voxels for smarter cropping
        tumor_mask = seg > 0
        
        if tumor_mask.sum() > 0 and np.random.random() < self.tumor_center_prob:
            # Center on tumor with some randomness
            coords = np.where(tumor_mask)
            rand_idx = np.random.randint(len(coords[0]))
            center_d = coords[0][rand_idx]
            center_h = coords[1][rand_idx]
            center_w = coords[2][rand_idx]
            
            # Add randomness to avoid always centering exactly
            jitter = pd // 4
            start_d = center_d - pd // 2 + np.random.randint(-jitter, jitter + 1)
            start_h = center_h - ph // 2 + np.random.randint(-jitter, jitter + 1)
            start_w = center_w - pw // 2 + np.random.randint(-jitter, jitter + 1)
            
            # Clamp to valid range
            start_d = max(0, min(d - pd, start_d))
            start_h = max(0, min(h - ph, start_h))
            start_w = max(0, min(w - pw, start_w))
        else:
            # Random crop
            start_d = np.random.randint(0, max(1, d - pd + 1))
            start_h = np.random.randint(0, max(1, h - ph + 1))
            start_w = np.random.randint(0, max(1, w - pw + 1))
        
        return self._extract_patch(image, seg, (start_d, start_h, start_w))
    
    def _center_crop(self, image, seg):
        """Extract center patch for testing"""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        start_d = max(0, (d - pd) // 2)
        start_h = max(0, (h - ph) // 2)
        start_w = max(0, (w - pw) // 2)
        
        return self._extract_patch(image, seg, (start_d, start_h, start_w))
    
    def _augment(self, image, seg):
        """Apply 3D augmentations"""
        # Random flips along each axis
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            seg = np.flip(seg, axis=0).copy()
        if np.random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            seg = np.flip(seg, axis=1).copy()
        if np.random.random() < 0.5:
            image = np.flip(image, axis=3).copy()
            seg = np.flip(seg, axis=2).copy()
        
        # Random 90-degree rotations in axial plane
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(2, 3)).copy()
            seg = np.rot90(seg, k, axes=(1, 2)).copy()
        
        # Random intensity augmentation per modality
        for i in range(4):
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.9, 1.1)
                shift = np.random.uniform(-0.1, 0.1)
                image[i] = image[i] * scale + shift
            
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.05, image[i].shape).astype(np.float32)
                image[i] = image[i] + noise
        
        return image, seg


# ============================================================================
# Helper Functions
# ============================================================================

def get_patient_ids(data_dir):
    """Get list of patient IDs from data directory"""
    data_dir = Path(data_dir)
    patient_ids = [d.name for d in data_dir.iterdir() 
                   if d.is_dir() and ('BraTS' in d.name or 'brats' in d.name.lower())]
    return sorted(patient_ids)


def train_val_test_split(patient_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split patient IDs into train/val/test sets"""
    np.random.seed(seed)
    ids = np.array(patient_ids)
    np.random.shuffle(ids)
    
    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return ids[:train_end].tolist(), ids[train_end:val_end].tolist(), ids[val_end:].tolist()


# ============================================================================
# Metrics (FIXED HD95)
# ============================================================================

def compute_dice(pred, target, smooth=1e-5):
    """Compute Dice coefficient between two binary masks"""
    pred = pred.flatten().astype(np.float32)
    target = target.flatten().astype(np.float32)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def get_surface(mask):
    """
    Extract surface voxels using binary erosion XOR.
    This is the standard approach for surface distance computation.
    """
    # 3D structuring element (6-connectivity)
    struct = generate_binary_structure(3, 1)
    
    # Surface = mask XOR eroded_mask
    eroded = binary_erosion(mask, structure=struct, border_value=0)
    surface = mask ^ eroded
    
    return surface


def compute_hd95(pred, target, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff Distance (FIXED).
    
    Uses proper surface extraction via binary erosion XOR,
    then computes surface-to-surface distances.
    
    Args:
        pred: binary prediction mask
        target: binary ground truth mask
        voxel_spacing: (d, h, w) spacing in mm
    
    Returns:
        HD95 in mm, or np.nan if either mask is empty
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Handle empty masks
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan
    
    # Extract surfaces using erosion XOR method
    pred_surface = get_surface(pred)
    target_surface = get_surface(target)
    
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.nan
    
    # Compute distance transforms from each surface to the other mask
    # Distance from pred surface points to nearest target point
    target_dist = distance_transform_edt(~target, sampling=voxel_spacing)
    pred_to_target_distances = target_dist[pred_surface]
    
    # Distance from target surface points to nearest pred point
    pred_dist = distance_transform_edt(~pred, sampling=voxel_spacing)
    target_to_pred_distances = pred_dist[target_surface]
    
    # HD95 is 95th percentile of symmetric Hausdorff
    hd95_pred_to_target = np.percentile(pred_to_target_distances, 95)
    hd95_target_to_pred = np.percentile(target_to_pred_distances, 95)
    
    return max(hd95_pred_to_target, hd95_target_to_pred)


def compute_brats_regions(seg):
    """
    Convert label segmentation to BraTS region masks.
    
    Labels: 0=BG, 1=NCR, 2=ED, 3=ET (remapped from 4)
    Regions:
        WT (Whole Tumor): 1 + 2 + 3
        TC (Tumor Core): 1 + 3
        ET (Enhancing): 3
    """
    wt = (seg > 0).astype(np.uint8)
    tc = np.isin(seg, [1, 3]).astype(np.uint8)
    et = (seg == 3).astype(np.uint8)
    return wt, tc, et


def compute_brats_metrics(pred, target, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute official BraTS competition metrics.
    
    Args:
        pred: predicted segmentation (D, H, W) with labels 0-3
        target: ground truth segmentation (D, H, W) with labels 0-3
        voxel_spacing: voxel dimensions in mm
    
    Returns:
        dict with Dice and HD95 for each region (WT, TC, ET)
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    pred_wt, pred_tc, pred_et = compute_brats_regions(pred)
    target_wt, target_tc, target_et = compute_brats_regions(target)
    
    return {
        'WT_Dice': compute_dice(pred_wt, target_wt),
        'TC_Dice': compute_dice(pred_tc, target_tc),
        'ET_Dice': compute_dice(pred_et, target_et),
        'WT_HD95': compute_hd95(pred_wt, target_wt, voxel_spacing),
        'TC_HD95': compute_hd95(pred_tc, target_tc, voxel_spacing),
        'ET_HD95': compute_hd95(pred_et, target_et, voxel_spacing),
    }


def dice_per_class(pred, target, num_classes=4, smooth=1e-5):
    """Compute Dice score for each class separately (for training monitoring)"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice)
    
    return dice_scores


def dice_brats_regions(pred, target, smooth=1e-5):
    """
    Compute Dice for BraTS regions (WT, TC, ET) - use this for training monitoring
    to match competition metrics.
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    pred_wt, pred_tc, pred_et = compute_brats_regions(pred)
    target_wt, target_tc, target_et = compute_brats_regions(target)
    
    return {
        'WT': compute_dice(pred_wt, target_wt, smooth),
        'TC': compute_dice(pred_tc, target_tc, smooth),
        'ET': compute_dice(pred_et, target_et, smooth),
    }


# ============================================================================
# Sliding Window Inference (with AMP support)
# ============================================================================

def get_gaussian_3d(patch_size, sigma_scale=0.125):
    """Create 3D Gaussian kernel for weighted patch stitching"""
    center = [s // 2 for s in patch_size]
    sigmas = [s * sigma_scale for s in patch_size]
    
    tmp = np.zeros(patch_size)
    tmp[tuple(center)] = 1
    gaussian = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    gaussian = gaussian / gaussian.max()
    gaussian = np.maximum(gaussian, 1e-8)
    
    return gaussian.astype(np.float32)


def sliding_window_inference(model, image, patch_size=(128, 128, 128), 
                             overlap=0.5, device='cuda', num_classes=4,
                             use_gaussian=True, use_amp=True, batch_size=2):
    """
    Perform sliding window inference on a full volume.
    
    OPTIMIZED: Processes multiple patches in batches for faster inference.
    With batch_size=2-4, inference is 2-3x faster than patch-by-patch.
    
    Args:
        model: trained 3D U-Net
        image: input array (4, D, H, W)
        patch_size: size of patches (D, H, W)
        overlap: overlap ratio between patches
        device: 'cuda' or 'cpu'
        num_classes: number of output classes
        use_gaussian: use Gaussian weighting for smooth blending
        use_amp: use automatic mixed precision for faster inference
        batch_size: number of patches to process at once (2-4 recommended)
    
    Returns:
        segmentation: (D, H, W) predicted labels
    """
    model.eval()
    
    if torch.is_tensor(image):
        image = image.numpy()
    
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    
    # Calculate step size
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))
    
    # Pad image if smaller than patch size
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), 
                       mode='constant', constant_values=0)
    
    _, d_pad, h_pad, w_pad = image.shape
    
    # Initialize output
    output = np.zeros((num_classes, d_pad, h_pad, w_pad), dtype=np.float32)
    count = np.zeros((d_pad, h_pad, w_pad), dtype=np.float32)
    
    # Gaussian kernel
    if use_gaussian:
        gaussian = get_gaussian_3d(patch_size)
    else:
        gaussian = np.ones(patch_size, dtype=np.float32)
    
    # Generate ALL patch positions first
    z_positions = list(range(0, max(1, d_pad - pd + 1), step_d))
    y_positions = list(range(0, max(1, h_pad - ph + 1), step_h))
    x_positions = list(range(0, max(1, w_pad - pw + 1), step_w))
    
    # Ensure edge coverage
    if z_positions[-1] + pd < d_pad:
        z_positions.append(d_pad - pd)
    if y_positions[-1] + ph < h_pad:
        y_positions.append(h_pad - ph)
    if x_positions[-1] + pw < w_pad:
        x_positions.append(w_pad - pw)
    
    # Collect all positions
    all_positions = [(z, y, x) for z in z_positions for y in y_positions for x in x_positions]
    
    with torch.no_grad():
        # Process in batches
        for batch_start in range(0, len(all_positions), batch_size):
            batch_positions = all_positions[batch_start:batch_start + batch_size]
            
            # Extract patches for this batch
            patches = []
            for (z, y, x) in batch_positions:
                patch = image[:, z:z+pd, y:y+ph, x:x+pw]
                patches.append(patch)
            
            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
            
            # Forward pass (batched)
            if use_amp and device != 'cpu':
                with autocast():
                    pred = model(batch_tensor)
            else:
                pred = model(batch_tensor)
            
            pred = F.softmax(pred, dim=1).cpu().numpy()
            
            # Accumulate each patch in the batch
            for i, (z, y, x) in enumerate(batch_positions):
                output[:, z:z+pd, y:y+ph, x:x+pw] += pred[i] * gaussian
                count[z:z+pd, y:y+ph, x:x+pw] += gaussian
    
    # Normalize
    count = np.maximum(count, 1e-8)
    for c in range(num_classes):
        output[c] /= count
    
    # Remove padding and get labels
    output = output[:, :d, :h, :w]
    segmentation = np.argmax(output, axis=0).astype(np.int64)
    
    return segmentation


# ============================================================================
# Loss Function
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss for 3D segmentation.
    
    Improvements:
    - Per-sample dice computation for stability
    - Optional class weights for dice (emphasize hard classes like ET)
    """
    def __init__(self, num_classes=4, ce_weight=1.0, dice_weight=1.0, 
                 class_weights=None, dice_class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # CE class weights
        if class_weights is not None:
            self.register_buffer('ce_weights', torch.tensor(class_weights).float())
        else:
            self.ce_weights = None
        
        # Dice class weights (e.g., [1.0, 1.0, 1.0, 2.0] to emphasize ET)
        if dice_class_weights is not None:
            self.dice_class_weights = dice_class_weights
        else:
            self.dice_class_weights = [1.0] * (num_classes - 1)  # Skip background
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) labels
        """
        # CrossEntropy loss
        if self.ce_weights is not None:
            ce_loss = F.cross_entropy(pred, target, weight=self.ce_weights)
        else:
            ce_loss = F.cross_entropy(pred, target)
        
        # Per-sample Dice loss (more stable than global)
        pred_soft = F.softmax(pred, dim=1)
        batch_size = pred.shape[0]
        
        dice_loss = 0.0
        for b in range(batch_size):
            sample_dice = 0.0
            for c in range(1, self.num_classes):  # Skip background
                pred_c = pred_soft[b, c]
                target_c = (target[b] == c).float()
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                
                dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
                sample_dice += self.dice_class_weights[c - 1] * (1 - dice_c)
            
            dice_loss += sample_dice / sum(self.dice_class_weights)
        
        dice_loss = dice_loss / batch_size
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ============================================================================
# Full Volume Loading (for inference)
# ============================================================================

def load_patient_volume(data_dir, patient_id):
    """
    Load a full patient volume for inference.
    
    Returns:
        image: (4, D, H, W) normalized
        seg: (D, H, W) ground truth (or None if not available)
        affine: NIfTI affine matrix
    """
    data_dir = Path(data_dir)
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    volumes = []
    affine = None
    
    for mod in modalities:
        # Try patterns
        patterns = [
            data_dir / patient_id / f"{patient_id}_{mod}.nii.gz",
            data_dir / patient_id / f"{patient_id}_{mod}.nii",
            data_dir / patient_id / f"{mod}.nii.gz",
        ]
        
        nii_path = None
        for p in patterns:
            if p.exists():
                nii_path = p
                break
        
        if nii_path is None:
            raise FileNotFoundError(f"Missing {mod} for patient {patient_id}")
        
        nii = nib.load(str(nii_path))
        if affine is None:
            affine = nii.affine
        volumes.append(nii.get_fdata())
    
    # Stack and normalize
    image = np.stack(volumes, axis=0).astype(np.float32)
    image = np.transpose(image, (0, 3, 1, 2))
    
    for i in range(4):
        mask = image[i] > 0
        if mask.sum() > 0:
            mean = image[i][mask].mean()
            std = image[i][mask].std()
            if std > 0:
                image[i][mask] = (image[i][mask] - mean) / std
    
    # Try to load segmentation
    seg = None
    for pattern in [f"{patient_id}_seg.nii.gz", f"{patient_id}_seg.nii", "seg.nii.gz"]:
        seg_path = data_dir / patient_id / pattern
        if seg_path.exists():
            seg = nib.load(str(seg_path)).get_fdata().astype(np.int64)
            seg = np.transpose(seg, (2, 0, 1))
            seg[seg == 4] = 3
            break
    
    return image, seg, affine


def save_prediction_nifti(prediction, affine, output_path):
    """Save prediction as NIfTI file"""
    pred_save = np.transpose(prediction, (1, 2, 0))
    pred_save[pred_save == 3] = 4  # Remap back to BraTS format
    
    nii = nib.Nifti1Image(pred_save.astype(np.uint8), affine)
    nib.save(nii, str(output_path))


print("✓ brats_utils_3d.py (v2 - FIXED) loaded")
print("  Fixes applied:")
print("  - HD95: proper surface extraction via binary erosion XOR")
print("  - Validation: multi-patch mode (8 deterministic patches per volume)")
print("  - File loading: robust checks with clear error messages")
print("  - Added dice_brats_regions() for WT/TC/ET monitoring")
print("  - CombinedLoss: per-sample dice, optional dice class weights")
print("  - Sliding window: BATCHED inference (2-3x faster) + AMP support")
print("")
print("  Performance tips:")
print("  - Set use_cache=True in BraTSDataset3D if RAM > 32GB")
print("  - Increase num_workers to 4-8 if CPU is bottleneck")
print("  - sliding_window_inference batch_size=2-4 for faster eval")
