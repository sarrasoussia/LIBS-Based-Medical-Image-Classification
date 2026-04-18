#!/usr/bin/env python3
"""
Comprehensive Experiment Results Report
"""
import json
from pathlib import Path

results_dir = Path("experiments/results")

def print_section(title):
    print("\n" + "=" * 120)
    print(f"  {title}")
    print("=" * 120)

def load_comparison_data(dataset_name):
    filepath = results_dir / dataset_name / "comparison_summary.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None

# Main report
print("\n")
print("█" * 120)
print("█" + " " * 118 + "█")
print("█" + "  COMPREHENSIVE EXPERIMENTS RESULTS REPORT".center(118) + "█")
print("█" + " " * 118 + "█")
print("█" * 120)

datasets = ["pathmnist", "bloodmnist"]

# Load all data
data = {}
for dataset in datasets:
    data[dataset] = load_comparison_data(dataset)

# Determine which models are available in each dataset
available_models = {}
for dataset in datasets:
    if data[dataset]:
        available_models[dataset] = sorted(data[dataset].keys())
    else:
        available_models[dataset] = []

# ============================================================================
# PATHMNIST RESULTS
# ============================================================================
print_section("PATHMNIST RESULTS")

if data["pathmnist"]:
    pathmnist_data = data["pathmnist"]
    print("\nModel Performance Metrics (PathMNIST - Test Set)")
    print("-" * 120)
    print(f"{'Model':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Macro':<15} {'ECE':<15}")
    print("-" * 120)
    
    for model in available_models["pathmnist"]:
        if model in pathmnist_data:
            d = pathmnist_data[model]
            print(f"{model:<25} {d['accuracy']:<15.4f} {d['precision_macro']:<15.4f} {d['recall_macro']:<15.4f} {d['f1_macro']:<15.4f} {d['ece']:<15.4f}")
    
    # Rankings
    print("\nRanking by Accuracy (PathMNIST):")
    print("-" * 70)
    rankings = sorted([(m, pathmnist_data[m]['accuracy']) for m in available_models["pathmnist"]], 
                      key=lambda x: x[1], reverse=True)
    for i, (model, acc) in enumerate(rankings, 1):
        gap_to_best = (acc - rankings[0][1]) * 100
        gap_to_worst = (acc - rankings[-1][1]) * 100
        print(f"  {i}. {model:<25} {acc:.4f}  (gap to best: {gap_to_best:+.2f}pp, to worst: {gap_to_worst:+.2f}pp)")

# ============================================================================
# BLOODMNIST RESULTS
# ============================================================================
print_section("BLOODMNIST RESULTS")

if data["bloodmnist"]:
    bloodmnist_data = data["bloodmnist"]
    print("\nModel Performance Metrics (BloodMNIST - Test Set)")
    print("-" * 120)
    print(f"{'Model':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Macro':<15} {'ECE':<15}")
    print("-" * 120)
    
    for model in available_models["bloodmnist"]:
        if model in bloodmnist_data:
            d = bloodmnist_data[model]
            print(f"{model:<25} {d['accuracy']:<15.4f} {d['precision_macro']:<15.4f} {d['recall_macro']:<15.4f} {d['f1_macro']:<15.4f} {d['ece']:<15.4f}")
    
    # Rankings
    print("\nRanking by Accuracy (BloodMNIST):")
    print("-" * 70)
    rankings = sorted([(m, bloodmnist_data[m]['accuracy']) for m in available_models["bloodmnist"]], 
                      key=lambda x: x[1], reverse=True)
    for i, (model, acc) in enumerate(rankings, 1):
        gap_to_best = (acc - rankings[0][1]) * 100
        gap_to_worst = (acc - rankings[-1][1]) * 100
        print(f"  {i}. {model:<25} {acc:.4f}  (gap to best: {gap_to_best:+.2f}pp, to worst: {gap_to_worst:+.2f}pp)")

# ============================================================================
# CROSS-DATASET ANALYSIS
# ============================================================================
print_section("CROSS-DATASET ANALYSIS")

if data["pathmnist"] and data["bloodmnist"]:
    pathmnist_data = data["pathmnist"]
    bloodmnist_data = data["bloodmnist"]
    
    print("\nAccuracy Comparison Across Datasets:")
    print("-" * 120)
    print(f"{'Model':<25} {'PathMNIST':<20} {'BloodMNIST':<20} {'Average':<20} {'Difference':<20}")
    print("-" * 120)
    
    common_models = set(available_models["pathmnist"]) & set(available_models["bloodmnist"])
    for model in sorted(common_models):
        if model in pathmnist_data and model in bloodmnist_data:
            path_acc = pathmnist_data[model]["accuracy"]
            blood_acc = bloodmnist_data[model]["accuracy"]
            avg_acc = (path_acc + blood_acc) / 2
            diff = abs(path_acc - blood_acc)
            print(f"{model:<25} {path_acc:<20.4f} {blood_acc:<20.4f} {avg_acc:<20.4f} {diff:<20.4f}")

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print_section("COMPARATIVE ANALYSIS")

if data["pathmnist"] and data["bloodmnist"]:
    pathmnist_data = data["pathmnist"]
    bloodmnist_data = data["bloodmnist"]
    
    print("\n1. ARCHITECTURE COMPARISON (Baseline CNN vs DenseNet121)")
    print("-" * 120)
    for dataset_name, data_dict in [("PathMNIST", pathmnist_data), ("BloodMNIST", bloodmnist_data)]:
        if "baseline_cnn" in data_dict and "densenet121" in data_dict:
            baseline_acc = data_dict["baseline_cnn"]["accuracy"]
            densenet_acc = data_dict["densenet121"]["accuracy"]
            improvement = (densenet_acc - baseline_acc) * 100
            print(f"  {dataset_name:<20} Baseline: {baseline_acc:.4f} → DenseNet121: {densenet_acc:.4f}  ({improvement:+.2f}pp pretrained advantage)")
    
    print("\n2. LIBS CONTRIBUTION (Raw CNN vs LIBS-CNN)")
    print("-" * 120)
    if "libs_cnn" in pathmnist_data and "baseline_cnn" in pathmnist_data:
        baseline_acc = pathmnist_data["baseline_cnn"]["accuracy"]
        libs_acc = pathmnist_data["libs_cnn"]["accuracy"]
        improvement = (libs_acc - baseline_acc) * 100
        print(f"  {'PathMNIST':<20} Baseline CNN: {baseline_acc:.4f} → LIBS-CNN: {libs_acc:.4f}  ({improvement:+.2f}pp LIBS contribution)")
    else:
        print(f"  {'BloodMNIST':<20} LIBS-CNN not available (old GA models in results)")
    
    print("\n3. LIBS WITH PRETRAINED (DenseNet121 vs LIBS-DenseNet121)")
    print("-" * 120)
    if "libs_densenet121" in pathmnist_data and "densenet121" in pathmnist_data:
        dense_acc = pathmnist_data["densenet121"]["accuracy"]
        libs_dense_acc = pathmnist_data["libs_densenet121"]["accuracy"]
        improvement = (libs_dense_acc - dense_acc) * 100
        print(f"  {'PathMNIST':<20} DenseNet121: {dense_acc:.4f} → LIBS-DenseNet121: {libs_dense_acc:.4f}  ({improvement:+.2f}pp LIBS contribution)")
    else:
        print(f"  {'BloodMNIST':<20} LIBS-DenseNet121 not available (old GA models in results)")

# ============================================================================
# KEY FINDINGS
# ============================================================================
print_section("KEY FINDINGS & INSIGHTS")

if data["pathmnist"] and data["bloodmnist"]:
    pathmnist_data = data["pathmnist"]
    bloodmnist_data = data["bloodmnist"]
    
    # Best overall model
    print("\n📊 OVERALL PERFORMANCE RANKING (Common Models):")
    all_accs = {}
    common_models = set(available_models["pathmnist"]) & set(available_models["bloodmnist"])
    for model in sorted(common_models):
        if model in pathmnist_data and model in bloodmnist_data:
            avg = (pathmnist_data[model]["accuracy"] + bloodmnist_data[model]["accuracy"]) / 2
            all_accs[model] = avg
    
    sorted_models = sorted(all_accs.items(), key=lambda x: x[1], reverse=True)
    for i, (model, avg_acc) in enumerate(sorted_models, 1):
        print(f"  {i}. {model:<25} Average: {avg_acc:.4f}")
    
    # Pretrained effectiveness
    print("\n🔍 PRETRAINED MODEL EFFECTIVENESS:")
    for dataset_name, data_dict in [("PathMNIST", pathmnist_data), ("BloodMNIST", bloodmnist_data)]:
        if "baseline_cnn" in data_dict and "densenet121" in data_dict:
            pretrained_gain = (data_dict["densenet121"]["accuracy"] - data_dict["baseline_cnn"]["accuracy"]) * 100
            print(f"  • {dataset_name}: DenseNet121 outperforms Baseline CNN by {pretrained_gain:+.2f}pp")
    
    # LIBS effectiveness
    print("\n🔄 LIBS REPRESENTATION EFFECTIVENESS:")
    if "libs_cnn" in pathmnist_data and "baseline_cnn" in pathmnist_data:
        libs_cnn_impact_path = (pathmnist_data["libs_cnn"]["accuracy"] - pathmnist_data["baseline_cnn"]["accuracy"]) * 100
        print(f"  • LIBS-CNN on Baseline:")
        print(f"    - PathMNIST: {libs_cnn_impact_path:+.2f}pp")
    
    if "libs_densenet121" in pathmnist_data and "densenet121" in pathmnist_data:
        libs_dense_impact_path = (pathmnist_data["libs_densenet121"]["accuracy"] - pathmnist_data["densenet121"]["accuracy"]) * 100
        print(f"  • LIBS-DenseNet121 on DenseNet:")
        print(f"    - PathMNIST: {libs_dense_impact_path:+.2f}pp")
    
    # Calibration analysis
    print("\n📏 CALIBRATION QUALITY (ECE - Expected Calibration Error):")
    print("  Lower ECE indicates better-calibrated confidence scores")
    for dataset_name, data_dict in [("PathMNIST", pathmnist_data), ("BloodMNIST", bloodmnist_data)]:
        dataset_key = dataset_name.lower()
        available = [m for m in available_models[dataset_key] if m in data_dict]
        if available:
            best_model = min([(m, data_dict[m]["ece"]) for m in available], key=lambda x: x[1])
            worst_model = max([(m, data_dict[m]["ece"]) for m in available], key=lambda x: x[1])
            print(f"  • {dataset_name}:")
            print(f"    - Best:  {best_model[0]:<25} ECE = {best_model[1]:.4f}")
            print(f"    - Worst: {worst_model[0]:<25} ECE = {worst_model[1]:.4f}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print_section("SUMMARY STATISTICS")

if data["pathmnist"] and data["bloodmnist"]:
    pathmnist_data = data["pathmnist"]
    bloodmnist_data = data["bloodmnist"]
    
    all_accuracies_path = [pathmnist_data[m]["accuracy"] for m in available_models["pathmnist"] if m in pathmnist_data]
    all_accuracies_blood = [bloodmnist_data[m]["accuracy"] for m in available_models["bloodmnist"] if m in bloodmnist_data]
    
    print("\nPathMNIST Statistics:")
    print("-" * 70)
    print(f"  Mean Accuracy:   {sum(all_accuracies_path)/len(all_accuracies_path):.4f}")
    print(f"  Max Accuracy:    {max(all_accuracies_path):.4f}")
    print(f"  Min Accuracy:    {min(all_accuracies_path):.4f}")
    print(f"  Range:           {(max(all_accuracies_path) - min(all_accuracies_path))*100:.2f}pp")
    
    print("\nBloodMNIST Statistics:")
    print("-" * 70)
    print(f"  Mean Accuracy:   {sum(all_accuracies_blood)/len(all_accuracies_blood):.4f}")
    print(f"  Max Accuracy:    {max(all_accuracies_blood):.4f}")
    print(f"  Min Accuracy:    {min(all_accuracies_blood):.4f}")
    print(f"  Range:           {(max(all_accuracies_blood) - min(all_accuracies_blood))*100:.2f}pp")
    
    print("\nNOTE: Results are expected to contain only baseline_cnn, densenet121, libs_cnn, libs_densenet121.")

print("\n" + "█" * 120)
print("█" + " " * 118 + "█")
print("█" + "  END OF REPORT".center(118) + "█")
print("█" + " " * 118 + "█")
print("█" * 120 + "\n")
