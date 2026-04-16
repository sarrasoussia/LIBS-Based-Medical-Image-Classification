import os
import glob
import json
from statistics import mean

with open("experiments/results/study/summary_statistics.json", "r", encoding="utf-8") as f:
    study = json.load(f)

fractions = ["0.25", "0.50", "0.75", "1.00"]

print("PathMNIST comparison (means)")
for frac in fractions:
    key = f"pathmnist|fraction={frac}"
    b = study[key]["baseline"]
    g = study[key]["ga_cnn"]

    root = f"experiments/results/study_pretrained/pathmnist/fraction_{frac}"
    d_acc, d_f1, d_ece = [], [], []
    gd_acc, gd_f1, gd_ece = [], [], []

    for seed_dir in sorted(glob.glob(os.path.join(root, "seed_*"))):
        d_path = os.path.join(seed_dir, "densenet121_metrics.json")
        gd_path = os.path.join(seed_dir, "ga_densenet121_metrics.json")

        if os.path.exists(d_path):
            m = json.load(open(d_path, "r", encoding="utf-8"))
            d_acc.append(m["accuracy"])
            d_f1.append(m["f1_macro"])
            d_ece.append(m["ece"])

        if os.path.exists(gd_path):
            m = json.load(open(gd_path, "r", encoding="utf-8"))
            gd_acc.append(m["accuracy"])
            gd_f1.append(m["f1_macro"])
            gd_ece.append(m["ece"])

    d_acc_m = mean(d_acc) if d_acc else None
    gd_acc_m = mean(gd_acc) if gd_acc else None
    d_f1_m = mean(d_f1) if d_f1 else None
    gd_f1_m = mean(gd_f1) if gd_f1 else None
    d_ece_m = mean(d_ece) if d_ece else None
    gd_ece_m = mean(gd_ece) if gd_ece else None

    print(f"\nfraction={frac} | n_dense={len(d_acc)} n_ga_dense={len(gd_acc)}")
    print(
        "  ACC baseline={:.4f} ga_cnn={:.4f} densenet={} ga_densenet={}".format(
            b["accuracy"]["mean"],
            g["accuracy"]["mean"],
            f"{d_acc_m:.4f}" if d_acc_m is not None else "NA",
            f"{gd_acc_m:.4f}" if gd_acc_m is not None else "NA",
        )
    )
    print(
        "  F1  baseline={:.4f} ga_cnn={:.4f} densenet={} ga_densenet={}".format(
            b["f1_macro"]["mean"],
            g["f1_macro"]["mean"],
            f"{d_f1_m:.4f}" if d_f1_m is not None else "NA",
            f"{gd_f1_m:.4f}" if gd_f1_m is not None else "NA",
        )
    )
    print(
        "  ECE densenet={} ga_densenet={}".format(
            f"{d_ece_m:.4f}" if d_ece_m is not None else "NA",
            f"{gd_ece_m:.4f}" if gd_ece_m is not None else "NA",
        )
    )
