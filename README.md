# Brain Tumor MRI Classification

This project builds a multiclass deep learning pipeline to classify brain MRI scans into:

- `glioma`
- `meningioma`
- `pituitary`
- `healthy`

The goal is to support early diagnosis and treatment planning by automating tumor classification from MRI images.

## Project Objective

Primary objective:

- Build and validate a high-performing model for four-class brain MRI classification.

Secondary objectives:

- Preprocess and split the dataset for reproducible experimentation.
- Apply augmentation to improve generalization.
- Compare multiple model architectures.
- Add explainability so model decisions are easier to interpret.
- Provide a lightweight dashboard to showcase outputs.

## Project Structure

```text
Brain-tumor-image-classification/
├── Dataset/
│   └── split_data/
│       ├── train/
│       ├── val/
│       └── test/
├── EDA.ipynb
├── Preprocessing_and_DataSplit.ipynb
├── Modeling_and_Training.ipynb
├── Explainability_and_Error_Analysis.ipynb
├── dashboard.py
├── results/
├── artifacts/
│   ├── models/
│   └── explainability/
└── README.md
```

## Recommended Notebook Order

Run the notebooks in this order:

1. `EDA.ipynb`
   - Explore the dataset
   - Check class distribution
   - Review image sizes and sample images

2. `Preprocessing_and_DataSplit.ipynb`
   - Download or load the dataset
   - Split into train, validation, and test sets
   - Preview augmentation

3. `Modeling_and_Training.ipynb`
   - Load the split dataset
   - Train the custom CNN baseline
   - Train the EfficientNetB0 transfer learning model
   - Save metrics, plots, and trained models

4. `Explainability_and_Error_Analysis.ipynb`
   - Load the saved best model
   - Generate test predictions
   - Review errors
   - Generate Grad-CAM visualizations

## Models Used

### 1. Custom CNN

A baseline convolutional neural network trained directly on the MRI dataset.

### 2. EfficientNetB0

A transfer learning model using a pretrained EfficientNetB0 backbone followed by a custom classification head.

## Evaluation Metrics

The modeling notebook reports:

- Accuracy
- Precision per class
- Recall per class
- F1-score per class
- Specificity per class
- Confusion matrix
- ROC-AUC using one-vs-rest multi-class evaluation

## Explainability

The explainability notebook uses Grad-CAM to visualize which image regions influenced the prediction.

Outputs include:

- Original MRI image
- Grad-CAM heatmap
- Overlay of the heatmap on the MRI image
- Error analysis for high-confidence mistakes

## Important Note About Dice and IoU

Dice and IoU require **tumor mask annotations**.

This project currently uses a classification dataset, so those metrics are only valid if you also have pixel-level tumor masks stored in a folder such as:

```text
Dataset/
└── tumor_masks/
    ├── glioma/
    ├── healthy/
    ├── meningioma/
    └── pituitary/
```

If mask annotations are not available, the notebook correctly skips Dice and IoU evaluation.

## How to Run

Open a terminal in the project folder:

```bash
cd "/Users/dnikita/Library/CloudStorage/OneDrive-athenahealth/Desktop/AI In Healthcare project/Brain-tumor-image-classification"
```

### Install dependencies

If needed, install notebook and dashboard dependencies:

```bash
uv pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow opencv-python streamlit
```

### Run the notebooks

Open the notebooks in VS Code or Jupyter and run them in order.

### Run the dashboard

After running the modeling and explainability notebooks:

```bash
uv run streamlit run dashboard.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dashboard Contents

The dashboard displays:

- Overall model metrics
- Per-class metrics
- Confusion matrix
- ROC curve plots
- Grad-CAM explainability outputs

## Saved Outputs

Expected saved outputs include:

- `results/custom_cnn/metrics.json`
- `results/custom_cnn/confusion_matrix.png`
- `results/custom_cnn/roc_curves.png`
- `results/efficientnet_b0/metrics.json`
- `results/efficientnet_b0/confusion_matrix.png`
- `results/efficientnet_b0/roc_curves.png`
- `artifacts/models/custom_cnn.keras`
- `artifacts/models/efficientnet_b0.keras`
- `artifacts/explainability/gradcam_examples.png`

## Literature Guidance

This project design was informed by the following papers:

- Frontiers in Computational Neuroscience (2024): brain tumor classification with a customized CNN approach
- Springer Nature (2025): deep learning and machine learning approaches for brain tumor MRI segmentation and classification

These papers helped guide:

- model comparison choices
- transfer learning use
- evaluation structure
- explainability emphasis

## Current Limitations

- The project currently assumes image-level class labels, not segmentation masks.
- Dice and IoU are not reported unless real tumor mask annotations are added.
- Grad-CAM provides qualitative explainability, not a clinical segmentation ground truth.

## Future Improvements

- Add true segmentation mask annotations
- Add clinical calibration analysis
- Add external validation dataset testing
- Add model version tracking and experiment logging
- Deploy a more polished dashboard for demonstration

## Author Workflow

This repository is organized to be notebook-first:

- notebooks for learning and step-by-step execution
- a simple Streamlit dashboard for presentation
- saved plots and models for reuse

This makes the project easier to present, explain, and extend.
