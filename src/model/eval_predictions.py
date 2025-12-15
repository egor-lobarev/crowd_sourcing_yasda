import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def _get_split_filenames(labels_df: pd.DataFrame, subset: str, seed: int):
    """Reproduce the same split used in training (70/15/15 with seed=42)."""
    if subset == "all":
        return set(labels_df["filename"].values)

    filenames = labels_df["filename"].values
    y = labels_df["label"].values

    train_fns, temp_fns, _, temp_y = train_test_split(
        filenames, y, test_size=0.3, random_state=seed, stratify=y
    )
    val_fns, test_fns, _, _ = train_test_split(
        temp_fns, temp_y, test_size=0.5, random_state=seed, stratify=temp_y
    )

    if subset == "train":
        return set(train_fns)
    if subset == "val":
        return set(val_fns)
    if subset == "test":
        return set(test_fns)
    return set(filenames)


def evaluate(pred_path: Path, labels_path: Path, subset: str = "all", seed: int = 42):
    # Load data
    preds_df = pd.read_csv(pred_path)
    labels_df = pd.read_csv(labels_path)

    # Merge on filename
    merged = preds_df.merge(labels_df, on="filename", how="inner")

    # Filter by subset if requested
    split_fns = _get_split_filenames(labels_df, subset, seed)
    merged = merged[merged["filename"].isin(split_fns)]

    total = len(merged)
    if total == 0:
        print("No overlapping filenames between predictions and labels.")
        return

    # Map predictions to numeric for comparison
    def to_num(pred):
        if pred == "TRUE":
            return 1
        if pred == "FALSE":
            return 0
        return None  # uncertain or unknown

    merged["pred_num"] = merged["prediction"].apply(to_num)

    # Confident predictions (exclude '?')
    confident = merged[merged["pred_num"].notnull()]
    confident_total = len(confident)

    correct_confident = (confident["pred_num"] == confident["label"]).sum()
    accuracy_confident = correct_confident / confident_total if confident_total > 0 else 0.0

    # Confusion matrix components (on confident only)
    tp = ((confident["pred_num"] == 1) & (confident["label"] == 1)).sum()
    tn = ((confident["pred_num"] == 0) & (confident["label"] == 0)).sum()
    fp = ((confident["pred_num"] == 1) & (confident["label"] == 0)).sum()
    fn = ((confident["pred_num"] == 0) & (confident["label"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Uncertain count
    uncertain_count = total - confident_total

    print("=" * 50)
    print("Evaluation Results")
    print(f"Total samples (with labels): {total}")
    print(f"Confident predictions: {confident_total}")
    print(f"Uncertain ('?') predictions: {uncertain_count}")
    print(f"Accuracy on confident predictions: {accuracy_confident:.4f}")
    print(f"Precision (confident): {precision:.4f}")
    print(f"Recall (confident):    {recall:.4f}")
    print(f"F1 (confident):        {f1:.4f}")
    if confident_total < total:
        coverage = confident_total / total
        print(f"Coverage (fraction of samples with confident predictions): {coverage:.4f}")
    print(f"\nConfusion (confident): TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 50)

    return {
        "total": total,
        "confident_total": confident_total,
        "uncertain_count": uncertain_count,
        "accuracy": accuracy_confident,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def plot_confusion(stats: dict, output_path: Path):
    """Plot confusion matrix heatmap for confident predictions."""
    data = [
        [stats["tn"], stats["fp"]],
        [stats["fn"], stats["tp"]],
    ]
    labels = [["TN", "FP"], ["FN", "TP"]]

    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(
        data,
        annot=[[f"{labels[i][j]}\n{data[i][j]}" for j in range(2)] for i in range(2)],
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["FALSE (0)", "TRUE (1)"])
    ax.set_yticklabels(["FALSE (0)", "TRUE (1)"])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV against ground-truth labels CSV")
    parser.add_argument(
        "--predictions",
        type=str,
        default="src/data/gt/pretrained_predictions.csv",
        help="Path to predictions CSV (must contain columns: filename, prediction)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="src/data/gt/labels.csv",
        help="Path to labels CSV (columns: filename, label)",
    )
    parser.add_argument(
        "--confusion_png",
        type=str,
        default="src/data/gt/confusion_heatmap.png",
        help="Path to save confusion heatmap (PNG)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Evaluate only on this split (reproduced with the same seed/stratify as training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproduction (must match training split)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    labels_path = Path(args.labels)

    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        return
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    stats = evaluate(pred_path, labels_path, subset=args.subset, seed=args.seed)

    # Plot confusion only if we have confident predictions
    if stats["confident_total"] > 0:
        plot_confusion(stats, Path(args.confusion_png))
    else:
        print("No confident predictions to plot confusion matrix.")


if __name__ == "__main__":
    main()

