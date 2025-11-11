# CMEF Melanoma Classification

We introduce CMEF, a contrastive‐learning framework that aligns clinical ABC criteria with Vision Transformer features to generate transparent, text‐based explanations while achieving expert‐level melanoma classification.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Data Source](#data-source)
7. [Data Preparation](#data-preparation)
8. [Usage](#usage)
   - [One-Command Execution](#one-command-execution)
   - [Step-by-Step Execution](#step-by-step-execution)
9. [Output and Results](#output-and-results)
10. [Contact and Support](#contact-and-support)

---

## Project Overview
This repository implements a two-stage melanoma classification workflow:

1. **Contrastive Learning Stage**: Train projection heads via a self-supervised NT-Xent loss to align image and clinical feature embeddings.
2. **Classification Stage**: Train separate image-based and clinical-based classifiers on frozen projection embeddings.

---

## Directory Structure

```
CMEF_melanoma_classification/
├── configs/                      
│   └── default.yaml             
├── data/                        
│   ├── Lesion_image/           
│   ├── Clinical_data/           
│   └── vit_model/               
│   └── dataset.py            
├── models/                      
│   ├── classifiers.py         
│   ├── feature_extractor.py  
│   └── projection.py           
├── utils/                       
│   ├── io.py                  
│   └── losses.py             
├── training/                   
│   ├── train_classifiers.py  
│   └── train_contrastive.py   
├── evaluation/                
│   └── evaluate.py              
├── scripts/                  
│   └── run_all.py              
├── requirements.txt            
└── README.md                    

```

---

## Prerequisites
- **Python**: 3.8+  
---

## Installation

1. **Clone** your anonymous repo (replace `<URL>`):  
   ```bash
   git clone <YOUR_REPO_URL>
   cd CMEF_melanoma_classification

2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration
**Edit** `configs/default.yaml` to match your local setup. The paths shown are examples:

```yaml
data:
  image_dir: "./data/Lesion image"
  train_csv: "./data/Clinical data/train_set_cleaned.csv"
  test_csv:  "./data/Clinical data/test_set_cleaned.csv"

model:
  image_model_path: "./data/vit_model/best_model.pth"
  output_dir:       "./run_result"

training:
  batch_size:            16
  lr:                    1e-3
  weight_decay:          1e-5
  epochs_contrastive:    15
  epochs_classifier:     10
  seed:                  999
  patience:              5
  temperature:           0.05
  projection_dim:        128
```

---

## Data Source (example)
We used a curated subset of public ISIC data: Total 3,578 training images (646 melanoma, 2,932 nevi)

---

## Data Preparation

1. **Provided Dataset**
   The dataset is located directly under the `data/` directory:

   ```bash
   data/Lesion/image/
   data/Clinical/data/train_set_cleaned.csv
   data/Clinical/data/test_set_cleaned.csv
   data/vit_model/best_model.pth
Currently, only 15 sample images are provided in `data/Lesion image/` for demonstration purposes.

2. **Using Your Own Dataset**  
To swap in a different dataset:
- Overwrite `data/Lesion image/` with your own images.  
- Overwrite `data/Clinical data/train_set_cleaned.csv` and `data/Clinical data/test_set_cleaned.csv` with your CSVs (must include `image_name` and `target` columns).  
- Place your pretrained ViT checkpoint at `data/vit_model/best_model.pth`, or update the path in `configs/default.yaml`.


3. **Update Configuration**  
After replacing the files, edit `configs/default.yaml` to match:
```yaml
data:
  image_dir: "./data/Lesion image"
  train_csv: "./data/Clinical data/train_set_cleaned.csv"
  test_csv:  "./data/Clinical data/test_set_cleaned.csv"
  ```
  
---

## Usage

### One-Command Execution

```bash
python scripts/run_all.py
```

This will:
1. Train projection heads (Stage 1)  
2. Train image & clinical classifiers (Stage 2)  

All outputs (checkpoints, logs, plots) will be saved under `output_dir`.

### Step-by-Step Execution

1. **Stage 1: Contrastive Training**
   ```bash
   python training/train_contrastive.py
   ```
2. **Stage 2: Classifier Training**
   ```bash
   python training/train_classifiers.py
   ```
3. **Evaluation**
   ```bash
   python evaluation/evaluate.py
   ```

---

## Output and Results
After training, `output_dir` will contain:
- **Model Checkpoints**:  
  - `projection_heads_final.pth`  
  - `image_classifier_final.pth`  
  - `clinical_classifier_final.pth`  
- **Logs**: Training logs in `.txt` files  
- **Figures**: Loss & AUC curves (`.png`) and confusion matrices
---

## Contact and Support
For issues, please open a GitHub Issue in this repository.