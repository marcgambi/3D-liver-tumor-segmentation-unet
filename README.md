# 3D-liver-tumor-segmentation-unet
3D U-Net deep learning pipeline for automatic liver tumor segmentation from volumetric medical images in NIFTI format using PyTorch and MONAI.

# Deep Learning Liver Tumor Segmentation

This project focuses on automatic liver tumor segmentation from volumetric medical images using a deep learning–based approach.

A 3D U-Net convolutional neural network was implemented and trained on medical imaging datasets in NIFTI format to perform tumor segmentation tasks.

---

## Project Overview

The pipeline includes:

- Medical image preprocessing
- Volumetric image resizing and normalization
- Custom dataset management
- 3D U-Net implementation
- Dice Loss optimization
- Model training and validation
- Segmentation result visualization

The project was developed using Python, PyTorch, MONAI, and medical imaging libraries.

---

## Technologies and Libraries

- Python
- PyTorch
- MONAI
- TorchIO
- NumPy
- Pandas
- NiBabel
- Matplotlib
- Scikit-learn

---

## Main Features

### Image Preprocessing
- Loading NIFTI medical images
- Volumetric resizing
- Tensor conversion
- Dataset preparation

### Deep Learning Model
- 3D U-Net architecture
- Dice Loss implementation
- Optimization with Adam optimizer
- GPU/CPU support

### Training Pipeline
- Training and validation loops
- Early stopping
- Loss tracking
- Model checkpoint saving

### Visualization
- Slice-by-slice segmentation visualization
- Comparison between:
  - input image
  - ground truth
  - predicted segmentation

---

## Example Workflow

1. Preprocess medical images
2. Create custom dataset and dataloaders
3. Train the 3D U-Net model
4. Validate segmentation performance
5. Visualize predicted tumor masks

---

## Skills Developed

- Deep Learning for Medical Imaging
- Biomedical Image Segmentation
- Volumetric Data Processing
- Python Programming
- Neural Network Training
- Experimental Data Analysis

---

## Notes

This project was developed for educational and research purposes in the biomedical engineering and medical imaging field.

---

## Dataset

The original medical imaging dataset is not included in this repository due to privacy and institutional data protection restrictions.
