# Applied AI in Biomedicine: Lung Nodule Malignancy Classification

**Team M2S**  
Mehrshad Alipoor, Shahryar Namdari Ghareghani, Maurizio Tirabassi  
May 14, 2025

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Data & Preprocessing](#data--preprocessing)  
4. [Modeling Approach](#modeling-approach)  
5. [Results](#results)  
6. [Interpretability](#interpretability)  
7. [Discussion & Future Work](#discussion--future-work)  
8. [Usage](#usage)  
9. [License](#license)  

---

## Overview

This project implements four deep‐learning classifiers for lung CT images:

- **Full-Slice 5-Class (“Full5”)**: Predict a malignancy score 1–5 on entire axial slices.  
- **Full-Slice Binary (“Full2”)**: Classify full slices as benign (1–3) vs. malignant (4–5).  
- **Nodule-Crop 5-Class (“Nod5”)**: Predict a malignancy score 1–5 on zoomed-in nodule crops.  
- **Nodule-Crop Binary (“Nod2”)**: Classify nodule crops as benign vs. malignant.  

Our pipeline uses pretrained CNN backbones (ConvNeXt-Tiny selected via screening) with tailored heads, extensive data augmentations, class-balanced sampling, and hyperparameter tuning.

---

## Repository Structure

