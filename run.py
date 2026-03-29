#!/bin/bash

echo "============================================================"
echo "BRAIN TUMOR CLASSIFICATION - COMPLETE PIPELINE"
echo "============================================================"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup Kaggle
echo "Setting up Kaggle..."
mkdir -p ~/.kaggle
# Note: You need to manually add kaggle.json file

# Run CNN model
echo ""
echo "============================================================"
echo "TRAINING CNN MODEL"
echo "============================================================"
python cnn_model.py --epochs 50 --batch_size 32

# Run EfficientNet model
echo ""
echo "============================================================"
echo "TRAINING EFFICIENTNET MODEL"
echo "============================================================"
python efficientnet_model.py --epochs 30 --fine_tune_epochs 10 --batch_size 32

# Compare models
echo ""
echo "============================================================"
echo "COMPARING MODELS"
echo "============================================================"
python compare_models.py

echo ""
echo "============================================================"
echo "ALL COMPLETE! Results saved in results/ directory"
echo "============================================================"
