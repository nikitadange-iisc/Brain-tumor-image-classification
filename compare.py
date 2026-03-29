"""
Compare CNN and EfficientNet models
Generates comprehensive comparison report
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_metrics(model_name):
    """Load metrics from JSON file"""
    path = f'results/{model_name}/all_metrics.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def create_comparison_report():
    """Generate comparison report between CNN and EfficientNet"""
    
    # Load metrics
    cnn_metrics = load_metrics('cnn')
    eff_metrics = load_metrics('efficientnet')
    
    if cnn_metrics is None or eff_metrics is None:
        print("Run both models first!")
        return
    
    # Create comparison directory
    os.makedirs('results/comparison', exist_ok=True)
    
    # 1. Accuracy Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Overall accuracy
    models = ['CNN', 'EfficientNet']
    accuracies = [cnn_metrics['classification_metrics']['accuracy'], 
                  eff_metrics['classification_metrics']['accuracy']]
    
    axes[0, 0].bar(models, accuracies, color=['blue', 'green'])
    axes[0, 0].set_title('Overall Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # Per-class F1-score comparison
    classes = list(cnn_metrics['classification_metrics'].keys())
    classes = [c for c in classes if c not in ['macro_avg', 'weighted_avg', 'accuracy']]
    
    cnn_f1 = [cnn_metrics['classification_metrics'][c]['f1-score'] for c in classes]
    eff_f1 = [eff_metrics['classification_metrics'][c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, cnn_f1, width, label='CNN', color='blue')
    axes[0, 1].bar(x + width/2, eff_f1, width, label='EfficientNet', color='green')
    axes[0, 1].set_xlabel('Tumor Classes')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('Per-Class F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # Segmentation metrics comparison
    metrics = ['Dice Coefficient', 'IoU (Jaccard)']
    cnn_seg = [cnn_metrics['segmentation_metrics']['dice_coefficient']['mean'],
               cnn_metrics['segmentation_metrics']['iou']['mean']]
    eff_seg = [eff_metrics['segmentation_metrics']['dice_coefficient']['mean'],
               eff_metrics['segmentation_metrics']['iou']['mean']]
    
    x = np.arange(len(metrics))
    axes[0, 2].bar(x - width/2, cnn_seg, width, label='CNN', color='blue')
    axes[0, 2].bar(x + width/2, eff_seg, width, label='EfficientNet', color='green')
    axes[0, 2].set_title('Segmentation Overlap Metrics', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(metrics)
    axes[0, 2].legend()
    axes[0, 2].set_ylim([0, 1])
    
    # Precision comparison
    axes[1, 0].bar(x - width/2, 
                   [cnn_metrics['classification_metrics'][c]['precision'] for c in classes],
                   width, label='CNN', color='blue')
    axes[1, 0].bar(x + width/2,
                   [eff_metrics['classification_metrics'][c]['precision'] for c in classes],
                   width, label='EfficientNet', color='green')
    axes[1, 0].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # Recall comparison
    axes[1, 1].bar(x - width/2,
                   [cnn_metrics['classification_metrics'][c]['recall'] for c in classes],
                   width, label='CNN', color='blue')
    axes[1, 1].bar(x + width/2,
                   [eff_metrics['classification_metrics'][c]['recall'] for c in classes],
                   width, label='EfficientNet', color='green')
    axes[1, 1].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    # Improvement percentage
    improvements = {}
    for class_name in classes:
        cnn_f1_val = cnn_metrics['classification_metrics'][class_name]['f1-score']
        eff_f1_val = eff_metrics['classification_metrics'][class_name]['f1-score']
        improvements[class_name] = ((eff_f1_val - cnn_f1_val) / cnn_f1_val) * 100
    
    axes[1, 2].bar(improvements.keys(), improvements.values(), color='purple')
    axes[1, 2].set_title('EfficientNet Improvement over CNN', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Improvement (%)')
    axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 2].set_xticklabels(classes, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/comparison/model_comparison.png', dpi=150)
    plt.close()
    
    # Generate text report
    report = f"""
    ============================================================
    BRAIN TUMOR CLASSIFICATION - MODEL COMPARISON REPORT
    ============================================================
    
    1. OVERALL ACCURACY
       CNN: {cnn_metrics['classification_metrics']['accuracy']:.4f} ({cnn_metrics['classification_metrics']['accuracy']*100:.2f}%)
       EfficientNet: {eff_metrics['classification_metrics']['accuracy']:.4f} ({eff_metrics['classification_metrics']['accuracy']*100:.2f}%)
       Improvement: {((eff_metrics['classification_metrics']['accuracy'] - cnn_metrics['classification_metrics']['accuracy']) / cnn_metrics['classification_metrics']['accuracy']) * 100:.1f}%
    
    2. PER-CLASS PERFORMANCE
    """
    
    for class_name in classes:
        report += f"""
    
    {class_name.upper()}:
       CNN - Precision: {cnn_metrics['classification_metrics'][class_name]['precision']:.4f}, 
             Recall: {cnn_metrics['classification_metrics'][class_name]['recall']:.4f}, 
             F1: {cnn_metrics['classification_metrics'][class_name]['f1-score']:.4f}
       EfficientNet - Precision: {eff_metrics['classification_metrics'][class_name]['precision']:.4f}, 
                     Recall: {eff_metrics['classification_metrics'][class_name]['recall']:.4f}, 
                     F1: {eff_metrics['classification_metrics'][class_name]['f1-score']:.4f}
       F1 Improvement: {improvements[class_name]:.1f}%
    """
    
    report += f"""
    
    3. SEGMENTATION OVERLAP METRICS
       Dice Coefficient:
         CNN: {cnn_metrics['segmentation_metrics']['dice_coefficient']['mean']:.4f} (+/- {cnn_metrics['segmentation_metrics']['dice_coefficient']['std']:.4f})
         EfficientNet: {eff_metrics['segmentation_metrics']['dice_coefficient']['mean']:.4f} (+/- {eff_metrics['segmentation_metrics']['dice_coefficient']['std']:.4f})
       
       IoU (Jaccard Index):
         CNN: {cnn_metrics['segmentation_metrics']['iou']['mean']:.4f} (+/- {cnn_metrics['segmentation_metrics']['iou']['std']:.4f})
         EfficientNet: {eff_metrics['segmentation_metrics']['iou']['mean']:.4f} (+/- {eff_metrics['segmentation_metrics']['iou']['std']:.4f})
    
    4. MACRO AVERAGES
       CNN:
         Precision: {cnn_metrics['classification_metrics']['macro_avg']['precision']:.4f}
         Recall: {cnn_metrics['classification_metrics']['macro_avg']['recall']:.4f}
         F1-Score: {cnn_metrics['classification_metrics']['macro_avg']['f1-score']:.4f}
       
       EfficientNet:
         Precision: {eff_metrics['classification_metrics']['macro_avg']['precision']:.4f}
         Recall: {eff_metrics['classification_metrics']['macro_avg']['recall']:.4f}
         F1-Score: {eff_metrics['classification_metrics']['macro_avg']['f1-score']:.4f}
    
    5. CONCLUSION
       {'EfficientNet outperforms CNN in all metrics' if eff_metrics['classification_metrics']['accuracy'] > cnn_metrics['classification_metrics']['accuracy'] else 'CNN performs better'}
       Recommended model: {'EfficientNet' if eff_metrics['classification_metrics']['accuracy'] > cnn_metrics['classification_metrics']['accuracy'] else 'CNN'}
    
    ============================================================
    """
    
    with open('results/comparison/comparison_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nComparison saved to results/comparison/")

if __name__ == '__main__':
    create_comparison_report()
