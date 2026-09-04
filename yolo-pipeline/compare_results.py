import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_results(output_dir):
    base_csv = os.path.join(output_dir, "results_baseline.csv")
    aug_csv = os.path.join(output_dir, "results_augmented.csv")

    if not os.path.exists(base_csv) or not os.path.exists(aug_csv):
        print(f"Error ")
        return

    df_base = pd.read_csv(base_csv)
    df_aug = pd.read_csv(aug_csv)

    base_mAP50_mean, base_mAP50_std = df_base['val_mAP50'].mean(), df_base['val_mAP50'].std()
    aug_mAP50_mean, aug_mAP50_std = df_aug['val_mAP50'].mean(), df_aug['val_mAP50'].std()

    base_mAP95_mean, base_mAP95_std = df_base['val_mAP50-95'].mean(), df_base['val_mAP50-95'].std()
    aug_mAP95_mean, aug_mAP95_std = df_aug['val_mAP50-95'].mean(), df_aug['val_mAP50-95'].std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({'font.size': 11})

    # 1. Matching mAP@50 for Fold
    x = np.arange(5)
    width = 0.35

    ax1.bar(x - width/2, df_base['val_mAP50'], width, label='Baseline (Real)', color='#4c72b0', alpha=0.9)
    ax1.bar(x + width/2, df_aug['val_mAP50'], width, label='Augmented (Real+Synth)', color='#55a868', alpha=0.9)
    ax1.set_ylabel('mAP@50')
    ax1.set_title('Matching mAP@50 for Single Fold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i}' for i in range(5)])
    ax1.set_ylim(0.7, 1.0)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Matching general metrics with error bars
    labels = ['mAP@50', 'mAP@50-95']
    base_means = [base_mAP50_mean, base_mAP95_mean]
    aug_means = [aug_mAP50_mean, aug_mAP95_mean]
    base_stds = [base_mAP50_std, base_mAP95_std]
    aug_stds = [aug_mAP50_std, aug_mAP95_std]

    x2 = np.arange(len(labels))
    ax2.bar(x2 - width/2, base_means, width, yerr=base_stds, capsize=5, label='Baseline', color='#4c72b0')
    ax2.bar(x2 + width/2, aug_means, width, yerr=aug_stds, capsize=5, label='Augmented', color='#55a868')
    ax2.set_ylabel('Score')
    ax2.set_title('Matching General Metrics with Error Bars')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.7, 1.0)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(labels)):
        diff = aug_means[i] - base_means[i]
        ax2.annotate(f"{diff:+.4f}",
                     xy=(x2[i] + width/2, aug_means[i] + aug_stds[i] + 0.01),
                     ha='center', va='bottom', fontweight='bold', color='#2b5c3b')

    plt.tight_layout()
    out_img = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(out_img, dpi=300)
    print(f"Comparison plot saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/content/outputs/yolo_experiments")
    args = parser.parse_args()
    compare_results(args.output_dir)
