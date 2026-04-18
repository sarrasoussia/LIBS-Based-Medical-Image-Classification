#!/usr/bin/env python3
"""
Per-Class Performance Analysis
"""
import json
from pathlib import Path
import statistics

results_dir = Path("experiments/results")

def load_metrics(dataset_name, model_name):
    filepath = results_dir / dataset_name / f"{model_name}_metrics.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None

print("\n" + "=" * 120)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 120)

# ============================================================================
# PATHMNIST PER-CLASS ANALYSIS
# ============================================================================
print("\n" + "█" * 120)
print("█ PATHMNIST PER-CLASS BREAKDOWN (9 Classes)")
print("█" * 120 + "\n")

models_pathmnist = ["baseline_cnn", "libs_cnn", "densenet121", "libs_densenet121"]

print("Class Performance - Recall (% of each class correctly identified):")
print("-" * 120)
print(f"{'Class':<15} {'Baseline CNN':<18} {'LIBS-CNN':<18} {'DenseNet121':<18} {'LIBS-DenseNet':<18}")
print("-" * 120)

for class_idx in range(9):
    class_names = ["Norm. Epithelium", "Norm. Stroma", "Tumor Epithelium", "Tumor Stroma", 
                   "Inflam.", "Necrotic/Debris", "Adipose", "Background", "Other"]
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
    
    recalls = []
    for model in models_pathmnist:
        metrics = load_metrics("pathmnist", model)
        if metrics:
            recall = metrics.get("recall_per_class", [])[class_idx] if class_idx < len(metrics.get("recall_per_class", [])) else 0
            recalls.append(f"{recall*100:>6.2f}%")
        else:
            recalls.append("N/A")
    
    print(f"{class_name:<15} {recalls[0]:<18} {recalls[1]:<18} {recalls[2]:<18} {recalls[3]:<18}")

print("\n" + "Class Performance - Precision (% of predictions correct for each class):")
print("-" * 120)
print(f"{'Class':<15} {'Baseline CNN':<18} {'LIBS-CNN':<18} {'DenseNet121':<18} {'LIBS-DenseNet':<18}")
print("-" * 120)

for class_idx in range(9):
    class_names = ["Norm. Epithelium", "Norm. Stroma", "Tumor Epithelium", "Tumor Stroma", 
                   "Inflam.", "Necrotic/Debris", "Adipose", "Background", "Other"]
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
    
    precisions = []
    for model in models_pathmnist:
        metrics = load_metrics("pathmnist", model)
        if metrics:
            precision = metrics.get("precision_per_class", [])[class_idx] if class_idx < len(metrics.get("precision_per_class", [])) else 0
            precisions.append(f"{precision*100:>6.2f}%")
        else:
            precisions.append("N/A")
    
    print(f"{class_name:<15} {precisions[0]:<18} {precisions[1]:<18} {precisions[2]:<18} {precisions[3]:<18}")

# Per-class difficulty analysis
print("\n" + "Per-Class Difficulty Analysis (Average Recall):")
print("-" * 70)

class_names = ["Norm. Epithelium", "Norm. Stroma", "Tumor Epithelium", "Tumor Stroma", 
               "Inflam.", "Necrotic/Debris", "Adipose", "Background", "Other"]
per_class_recalls = {}

for class_idx in range(9):
    recalls = []
    for model in models_pathmnist:
        metrics = load_metrics("pathmnist", model)
        if metrics and class_idx < len(metrics.get("recall_per_class", [])):
            recalls.append(metrics["recall_per_class"][class_idx])
    
    if recalls:
        avg_recall = statistics.mean(recalls)
        per_class_recalls[class_names[class_idx]] = avg_recall

sorted_classes = sorted(per_class_recalls.items(), key=lambda x: x[1], reverse=True)
print(f"{'Easiest Classes (High Recall)':^70}")
for i, (class_name, recall) in enumerate(sorted_classes[:3], 1):
    print(f"  {i}. {class_name:<30} {recall*100:>6.2f}% avg recall")

print(f"\n{'Hardest Classes (Low Recall)':^70}")
for i, (class_name, recall) in enumerate(sorted_classes[-3:], 1):
    print(f"  {i}. {class_name:<30} {recall*100:>6.2f}% avg recall")

# ============================================================================
# BLOODMNIST PER-CLASS ANALYSIS
# ============================================================================
print("\n" + "█" * 120)
print("█ BLOODMNIST PER-CLASS BREAKDOWN (8 Classes)")
print("█" * 120 + "\n")

models_bloodmnist = ["baseline_cnn", "densenet121"]

print("Class Performance - Recall (% of each class correctly identified):")
print("-" * 100)
print(f"{'Class':<20} {'Baseline CNN':<20} {'DenseNet121':<20}")
print("-" * 100)

class_names_blood = ["Basophils", "Eosinophils", "Lymphocytes", "Monocytes", 
                      "Neutrophils", "Platelet Clumps", "Red Blood Cells", "White Blood Cells"]

for class_idx in range(8):
    class_name = class_names_blood[class_idx] if class_idx < len(class_names_blood) else f"Class {class_idx}"
    
    recalls = []
    for model in models_bloodmnist:
        metrics = load_metrics("bloodmnist", model)
        if metrics:
            recall = metrics.get("recall_per_class", [])[class_idx] if class_idx < len(metrics.get("recall_per_class", [])) else 0
            recalls.append(f"{recall*100:>6.2f}%")
        else:
            recalls.append("N/A")
    
    print(f"{class_name:<20} {recalls[0]:<20} {recalls[1]:<20}")

print("\n" + "Class Performance - Precision (% of predictions correct for each class):")
print("-" * 100)
print(f"{'Class':<20} {'Baseline CNN':<20} {'DenseNet121':<20}")
print("-" * 100)

for class_idx in range(8):
    class_name = class_names_blood[class_idx] if class_idx < len(class_names_blood) else f"Class {class_idx}"
    
    precisions = []
    for model in models_bloodmnist:
        metrics = load_metrics("bloodmnist", model)
        if metrics:
            precision = metrics.get("precision_per_class", [])[class_idx] if class_idx < len(metrics.get("precision_per_class", [])) else 0
            precisions.append(f"{precision*100:>6.2f}%")
        else:
            precisions.append("N/A")
    
    print(f"{class_name:<20} {precisions[0]:<20} {precisions[1]:<20}")

# Per-class difficulty analysis
print("\n" + "Per-Class Difficulty Analysis (Average Recall across models):")
print("-" * 70)

per_class_recalls_blood = {}

for class_idx in range(8):
    recalls = []
    for model in models_bloodmnist:
        metrics = load_metrics("bloodmnist", model)
        if metrics and class_idx < len(metrics.get("recall_per_class", [])):
            recalls.append(metrics["recall_per_class"][class_idx])
    
    if recalls:
        avg_recall = statistics.mean(recalls)
        per_class_recalls_blood[class_names_blood[class_idx]] = avg_recall

sorted_classes_blood = sorted(per_class_recalls_blood.items(), key=lambda x: x[1], reverse=True)
print(f"{'Easiest Classes (High Recall)':^70}")
for i, (class_name, recall) in enumerate(sorted_classes_blood[:3], 1):
    print(f"  {i}. {class_name:<30} {recall*100:>6.2f}% avg recall")

print(f"\n{'Hardest Classes (Low Recall)':^70}")
for i, (class_name, recall) in enumerate(sorted_classes_blood[-3:], 1):
    print(f"  {i}. {class_name:<30} {recall*100:>6.2f}% avg recall")

print("\n" + "=" * 120 + "\n")
