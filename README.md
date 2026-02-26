# Glaucoma Disease Detection Web Application

A lightweight Flask application that combines advanced computer vision models to screen fundus images for glaucoma. The project showcases both **segmentation** and **classification** neural networks, provides explainable outputs (Grad-CAM & segmentation maps), and is structured for easy training, deployment, and extension.


## 🔍 Key Highlights

- **Dual‑model architecture**
  - **U‑Net** for optic disc/cup segmentation and CDR (Cup‑to‑Disc Ratio) calculation
  - **DenseNet‑121** classifier for glaucoma vs. normal prediction

- **Explainable AI**
  - Grad‑CAM visualizations show where the classifier focuses
  - Segmentation overlays illustrate the anatomical regions used for CDR

- **Web interface** (Flask + Tailwind CSS)
  - Simple image upload form
  - Immediate results with risk assessment and visual aids
  - Printable report page for easy sharing

- **Training scripts included**
  - `unet-cdr-g1020.py` – dataset loading, U‑Net training, mask post‑processing, CDR computation
  - `denseNet121.py` – data augmentation, custom focal loss, two‑phase training, evaluation

- **Deployment ready**
  - `requirements.txt` captures all dependencies (TensorFlow 2.10, Flask, OpenCV, etc.)
  - `runtime.txt` pins Python 3.10 for cloud platforms
  - `.gitignore` excludes heavy datasets and model artifacts


## 🛠️ What’s in the repository

```
app-backup2.py           # Main Flask application (inference + explainability)
unet-cdr-g1020.py        # U-Net segmentation training script
denseNet121.py           # DenseNet-121 classification training script
requirements.txt         # Python dependencies
runtime.txt              # Python version for deployment
templates/               # HTML views for upload and result pages
model/                   # Pretrained HDF5 model weights (tracked by Git)
.gitignore               # Excluded data directories & backups
``` 

> 📁 Large folders such as `data/`, `G1020/`, and `models/` are intentionally ignored by Git.


## 🎯 Usage Instructions

1. **Prepare environment**
   ```bash
   python -m venv venv       # create virtual environment
   venv\Scripts\Activate.ps1   # activate (Windows)
   pip install -r requirements.txt
   ```

2. **Train models** *(optional)*
   - Place cropped images/masks under `G1020/Images_Cropped/img` and `G1020/Masks_Cropped/img`.
   - Run `python unet-cdr-g1020.py` to train the segmentation network.
   - Structure classification data under `G1020_Structured/Train` and `G1020_Structured/Val` as `Glaucoma/` & `Normal/` subfolders.
   - Run `python denseNet121.py` to train the classifier.

3. **Run the app**
   ```bash
   python app-backup2.py
   ```
   Navigate to `http://localhost:5000` and upload a fundus image (JPG/PNG).

4. **View results**
   - CDR value and risk level (low, moderate, high)
   - Glaucoma classification with confidence
   - Segmentation and Grad‑CAM visualizations


## 📈 Why this project stands out

- **Medical imaging expertise** – handling fundus images and computing clinical metrics
- **Deep learning know‑how** – custom loss functions, two‑phase training, class weighting, data augmentation
- **Explainability focus** – implements both CAM and segmentation to justify predictions
- **Full‑stack delivery** – from dataset preprocessing to a polished web frontend

Whether you’re a recruiter, hiring manager, or another developer, this project demonstrates a strong blend of ML engineering and application development skills.


