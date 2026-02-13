# 3D Brain Tumor Segmentation (BraTS 2020)
**Deep Learning for Automated Glioma Detection in Multi-Modal MRI**  
Michael Banks and Sara Petrosian

## Overview
This project builds an end-to-end deep learning pipeline to segment gliomas from **multi-modal MRI** using U-Net architectures. The work iterates from baseline experiments through a full **3D U-Net** that performs volumetric inference.

## Problem
Gliomas are deadly brain tumors with a short median survival. Manual segmentation is slow, subjective, and difficult due to high variability in tumor appearance—yet accurate segmentation is critical for surgical planning.

## Dataset
**BraTS 2020 (369 patients)**  
- Modalities: **T1, T1ce, T2, FLAIR**
- Volume shape: **240×240×155**
- Split: **258 train / 55 val / 56 test**

## Model Evolution
Iterative development from baseline to higher performance:
- **Models 1–3:** Baseline exploration  
- **Model 4:** 2D U-Net, single modality  
- **Model 5:** 2D U-Net, 4 modalities  
- **Model 6:** **3D U-Net** — full volumetric processing

## Final Results (Dice)
- **Whole Tumor:** 89%  
- **Tumor Core:** 85%  
- **Enhancing Tumor:** 78%

## Key Techniques
- **U-Net architecture:** encoder–decoder with skip connections for precise boundaries  
- **3D convolutions:** volumetric filters capture tumor structure across slices  
- **Loss for class imbalance:** Dice + Cross-Entropy (handles heavy imbalance)  
- **Patch-based training:** memory-efficient training on 3D patches  
- **Sliding-window inference:** overlapping patches + blending for smooth predictions  

## Training Details
- Framework: PyTorch  
- GPU: NVIDIA A10G  
- Params: ~22.6M  
- Training time: ~6 hours

---

# Visualizations (go to Model 6)
For the best qualitative results quickly, jump to **Model 6** for:
- sample predictions
- raw MRI vs ground truth vs predicted masks
- final visualization panels

➡️ **Go to:** `Model6.ipynb` and scroll to the visualization section.
