"""
EfficientNet Model for Brain Tumor Classification
Includes: Classification metrics + Segmentation overlap analysis
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
import json
from datetime import datetime
import argparse

class EfficientNetBrainTumorModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, regularization_strength=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.regularization_strength = regularization_strength
        self.model = None
        self.history = None
        self.class_names = None
        
    def build_model(self):
        """Build EfficientNet architecture with transfer learning"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential(name="EfficientNet_Brain_Tumor_Classifier")
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        
        # Custom classification head
        model.add(layers.Dense(512, activation='relu', 
                              kernel_regularizer=l2(self.regularization_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(256, activation='relu',
                              kernel_regularizer=l2(self.regularization_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return model
    
    def fine_tune(self, unfreeze_layers=50):
        """Unfreeze some layers for fine-tuning"""
        # Unfreeze the last 'unfreeze_layers' layers
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze early layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                    keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
        )
        print(f"Fine-tuning with {unfreeze_layers} unfrozen layers")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                    keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
        )
        return self.model
    
    def train(self, train_generator, val_generator, epochs=30, fine_tune_epochs=10):
        # Phase 1: Train top layers only
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint('efficientnet_best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        print("Phase 1: Training top layers...")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        history1 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print("\nPhase 2: Fine-tuning...")
        self.fine_tune(unfreeze_layers=50)
        
        history2 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        return self.history

class EfficientNetMetricsCalculator:
    def __init__(self, model, test_generator, class_names):
        self.model = model
        self.test_generator = test_generator
        self.class_names = class_names
        self.predictions = None
        self.true_labels = None
        self.pred_classes = None
        
    def get_predictions(self):
        """Get model predictions"""
        self.predictions = self.model.predict(self.test_generator, verbose=0)
        self.pred_classes = np.argmax(self.predictions, axis=1)
        self.true_labels = self.test_generator.classes
        return self.predictions, self.pred_classes, self.true_labels
    
    def calculate_classification_metrics(self):
        """Calculate precision, recall, f1-score for each class"""
        report = classification_report(self.true_labels, self.pred_classes, 
                                      target_names=self.class_names, output_dict=True)
        
        metrics = {}
        for class_name in self.class_names:
            metrics[class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }
        
        metrics['macro_avg'] = {
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score']
        }
        
        metrics['weighted_avg'] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        }
        
        metrics['accuracy'] = report['accuracy']
        
        return metrics
    
    def plot_confusion_matrix(self, save_path='results/efficientnet/'):
        """Generate and save confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        cm = confusion_matrix(self.true_labels, self.pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   square=True, cbar_kws={"shrink": 0.8})
        plt.title('EfficientNet Model - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=150)
        plt.close()
        
        # Save as JSON
        cm_json = {
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names
        }
        with open(f'{save_path}/confusion_matrix.json', 'w') as f:
            json.dump(cm_json, f, indent=2)
        
        return cm
    
    def plot_roc_curves(self, save_path='results/efficientnet/'):
        """Plot ROC curves for multi-class classification"""
        os.makedirs(save_path, exist_ok=True)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(self.true_labels, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('EfficientNet Model - Multi-class ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=150)
        plt.close()
    
    def generate_heatmap(self, image, class_idx):
        """Generate Grad-CAM heatmap for segmentation overlap analysis"""
        # Get the last convolutional layer from EfficientNet
        base_model = self.model.layers[0]
        last_conv_layer = None
        
        for layer in reversed(base_model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
            loss = predictions[:, class_idx]
        
        # Get gradients and compute heatmap
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def create_synthetic_mask(self, image, threshold=0.5):
        """Create synthetic tumor mask from heatmap for segmentation analysis"""
        pred_class = np.argmax(self.predictions[0]) if len(self.predictions.shape) > 1 else self.pred_classes[0]
        heatmap = self.generate_heatmap(image, pred_class)
        
        if heatmap is None:
            return None
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create binary mask based on threshold
        binary_mask = (heatmap_resized > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5,5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    def calculate_segmentation_metrics(self, sample_images, sample_count=50):
        """Calculate Dice coefficient and IoU for segmentation overlap"""
        dice_scores = []
        iou_scores = []
        
        # Get sample batch
        sample_indices = np.random.choice(len(self.test_generator.filenames), 
                                         min(sample_count, len(self.test_generator.filenames)), 
                                         replace=False)
        
        for idx in sample_indices:
            # Get image and true label
            image_path = os.path.join(self.test_generator.directory, 
                                     self.test_generator.filenames[idx])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (224, 224)) / 255.0
            
            # Generate predicted mask from heatmap
            pred_mask = self.create_synthetic_mask(image_resized)
            
            if pred_mask is None:
                continue
            
            # Create synthetic ground truth mask (based on tumor presence)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, true_mask = cv2.threshold(gray, np.percentile(gray, 70), 1, cv2.THRESH_BINARY)
            true_mask = cv2.resize(true_mask, (pred_mask.shape[1], pred_mask.shape[0]))
            
            # Calculate Dice coefficient
            intersection = np.sum(pred_mask * true_mask)
            dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-7)
            dice_scores.append(dice)
            
            # Calculate IoU (Jaccard index)
            union = np.sum(pred_mask) + np.sum(true_mask) - intersection
            iou = intersection / (union + 1e-7)
            iou_scores.append(iou)
        
        return {
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'mean_iou': np.mean(iou_scores),
            'std_iou': np.std(iou_scores),
            'dice_scores': dice_scores,
            'iou_scores': iou_scores
        }
    
    def plot_segmentation_results(self, sample_count=5, save_path='results/efficientnet/'):
        """Visualize segmentation overlap results"""
        os.makedirs(save_path, exist_ok=True)
        
        sample_indices = np.random.choice(len(self.test_generator.filenames), 
                                         sample_count, replace=False)
        
        fig, axes = plt.subplots(sample_count, 4, figsize=(16, 4*sample_count))
        
        for row, idx in enumerate(sample_indices):
            # Load image
            image_path = os.path.join(self.test_generator.directory, 
                                     self.test_generator.filenames[idx])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (224, 224)) / 255.0
            
            # Get true label
            true_label = self.class_names[self.true_labels[idx]]
            pred_label = self.class_names[self.pred_classes[idx]]
            
            # Generate heatmap
            pred_class_idx = self.pred_classes[idx]
            heatmap = self.generate_heatmap(image_resized, pred_class_idx)
            
            if heatmap is not None:
                # Resize heatmap
                heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                
                # Create overlay
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
                
                # Generate mask
                pred_mask = self.create_synthetic_mask(image_resized)
                if pred_mask is not None:
                    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                                  (image.shape[1], image.shape[0]))
                    
                    # Create mask overlay
                    mask_overlay = image.copy()
                    mask_overlay[pred_mask_resized == 1] = [0, 255, 0]
                    
                    # Display images
                    axes[row, 0].imshow(image)
                    axes[row, 0].set_title(f'Original MRI\nTrue: {true_label}', fontsize=10)
                    axes[row, 0].axis('off')
                    
                    axes[row, 1].imshow(heatmap_resized, cmap='jet')
                    axes[row, 1].set_title(f'Attention Heatmap\nPred: {pred_label}', fontsize=10)
                    axes[row, 1].axis('off')
                    
                    axes[row, 2].imshow(overlay)
                    axes[row, 2].set_title('Overlay', fontsize=10)
                    axes[row, 2].axis('off')
                    
                    axes[row, 3].imshow(mask_overlay)
                    axes[row, 3].set_title(f'Tumor Segmentation Mask\nDice: {self.calculate_segmentation_metrics([image_resized], 1)["mean_dice"]:.3f}', fontsize=10)
                    axes[row, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/segmentation_visualization.png', dpi=150)
        plt.close()
    
    def save_all_metrics(self, save_path='results/efficientnet/'):
        """Save all calculated metrics to JSON"""
        os.makedirs(save_path, exist_ok=True)
        
        # Get all metrics
        classification_metrics = self.calculate_classification_metrics()
        seg_metrics = self.calculate_segmentation_metrics(None)
        
        all_metrics = {
            'model_type': 'EfficientNet',
            'classification_metrics': classification_metrics,
            'segmentation_metrics': {
                'dice_coefficient': {
                    'mean': float(seg_metrics['mean_dice']),
                    'std': float(seg_metrics['std_dice'])
                },
                'iou': {
                    'mean': float(seg_metrics['mean_iou']),
                    'std': float(seg_metrics['std_iou'])
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{save_path}/all_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet for Brain Tumor Classification')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--fine_tune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    print("="*60)
    print("EFFICIENTNET MODEL FOR BRAIN TUMOR CLASSIFICATION")
    print("="*60)
    
    # Download dataset
    print("Downloading dataset from Kaggle...")
    os.system('kaggle datasets download miadul/brain-tumor-mri-dataset')
    os.system('unzip -q brain-tumor-mri-dataset.zip')
    
    dataset_path = 'brain-tumor-mri-dataset/brain_tumor_dataset'
    
    # Get class names
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Classes: {classes}")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path, target_size=(224,224), batch_size=args.batch_size,
        class_mode='categorical', subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        dataset_path, target_size=(224,224), batch_size=args.batch_size,
        class_mode='categorical', subset='validation'
    )
    
    # Create and train EfficientNet
    eff_model = EfficientNetBrainTumorModel(num_classes=len(classes))
    model = eff_model.build_model()
    model = eff_model.compile_model()
    model.summary()
    
    history = eff_model.train(train_generator, val_generator, 
                             epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs)
    
    # Evaluate
    evaluator = EfficientNetMetricsCalculator(model, val_generator, classes)
    evaluator.get_predictions()
    
    # Generate all metrics
    classification_metrics = evaluator.calculate_classification_metrics()
    confusion_mat = evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curves()
    seg_metrics = evaluator.calculate_segmentation_metrics(None)
    evaluator.plot_segmentation_results()
    all_metrics = evaluator.save_all_metrics()
    
    # Print results
    print("\n" + "="*60)
    print("EFFICIENTNET MODEL - FINAL RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {classification_metrics['accuracy']:.4f}")
    print("\nPer-Class Metrics:")
    for class_name in classes:
        print(f"\n{class_name}:")
        print(f"  Precision: {classification_metrics[class_name]['precision']:.4f}")
        print(f"  Recall: {classification_metrics[class_name]['recall']:.4f}")
        print(f"  F1-Score: {classification_metrics[class_name]['f1-score']:.4f}")
    
    print("\nSegmentation Overlap Metrics:")
    print(f"  Mean Dice Coefficient: {seg_metrics['mean_dice']:.4f}")
    print(f"  Mean IoU (Jaccard): {seg_metrics['mean_iou']:.4f}")
    
    print("\nResults saved to results/efficientnet/")
    
    # Save training history plot
    if isinstance(history, dict):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('EfficientNet Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('EfficientNet Loss')
        plt.legend()
        plt.savefig('results/efficientnet/training_history.png')
        plt.close()

if __name__ == '__main__':
    main()
