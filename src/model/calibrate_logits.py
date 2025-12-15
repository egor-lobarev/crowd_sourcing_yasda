import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibrate_logits(
    pred_path: Path,
    labels_path: Path,
    output_path: Path,
):
    """Fit a logistic regression on logits -> labels (Platt scaling)."""
    preds_df = pd.read_csv(pred_path)
    labels_df = pd.read_csv(labels_path)

    merged = preds_df.merge(labels_df, on="filename", how="inner")

    if len(merged) == 0:
        print("No overlapping filenames between predictions and labels.")
        return

    X = merged["logit"].values.reshape(-1, 1)
    y = merged["label"].values.astype(int).values

    print(f"Fitting logistic regression on {len(X)} samples...")

    # Simple logistic regression (Platt scaling)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X, y)

    # Evaluate calibration quality
    probs_raw = 1 / (1 + np.exp(-X))  # original sigmoid
    probs_cal = clf.predict_proba(X)[:, 1]

    brier_raw = brier_score_loss(y, probs_raw)
    brier_cal = brier_score_loss(y, probs_cal)

    acc_raw = ((probs_raw > 0.5).astype(int) == y).mean()
    acc_cal = ((probs_cal > 0.5).astype(int) == y).mean()

    print("=" * 50)
    print("Calibration results (on GT set used for fitting):")
    print(f"Raw sigmoid  - Brier: {brier_raw:.4f}, Acc@0.5: {acc_raw:.4f}")
    print(f"Calibrated   - Brier: {brier_cal:.4f}, Acc@0.5: {acc_cal:.4f}")
    print("=" * 50)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Saved calibrated logistic regression to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate logits with logistic regression (Platt scaling)")
    parser.add_argument(
        "--predictions",
        type=str,
        default="src/data/gt/pretrained_predictions.csv",
        help="Path to predictions CSV with columns: filename, logit",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="src/data/gt/labels.csv",
        help="Path to labels CSV with columns: filename, label",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/model/logit_calibrator.pkl",
        help="Path to save calibrated logistic regression model",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        return
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    calibrate_logits(pred_path, labels_path, output_path)


if __name__ == "__main__":
    main()


