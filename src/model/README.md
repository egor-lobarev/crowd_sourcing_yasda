# Model Training, Calibration, Inference, Evaluation

This README collects the key commands and workflows for the model folder.

## 1) Train

Pretrained (recommended):
```bash
python src/model/train.py --model pretrained --model_name swin_tiny_patch4_window7_224 --img_size 128 --batch_size 16 --epochs 50
```

ConvNeXt-Tiny:
```bash
python src/model/train.py --model pretrained --model_name convnext_tiny.fb_in22k --img_size 128
```

UNet (baseline):
```bash
python src/model/train.py --model unet --img_size 256
```

**Continue training from the last fine-tuned checkpoint** (auto-loads if present):
```bash
python src/model/train.py --model pretrained --model_name convnext_tiny.fb_in22k --img_size 128 --lr 1e-5 --epochs 20
```

## 2) Calibration (Platt scaling on logits)

Calibrate logits on the labeled GT set:
```bash
python src/model/calibrate_logits.py \
  --predictions src/data/gt/pretrained_predictions.csv \
  --labels src/data/gt/labels.csv \
  --output src/model/logit_calibrator.pkl
```

Outputs: `logit_calibrator.pkl` and calibration metrics (Brier, Acc) before/after.

## 3) Inference (with optional calibration)

Predict on unlabeled images, mark uncertain as `?`:
```bash
python src/model/inference.py \
  --model_type pretrained \
  --model_name convnext_tiny.fb_in22k \
  --img_size 128 \
  --input_dir src/data/raw/images \
  --output_csv src/data/raw/predictions.csv \
  --uncertainty_threshold 0.25
```

Use calibrated probabilities:
```bash
python src/model/inference.py \
  --model_type pretrained \
  --model_name convnext_tiny.fb_in22k \
  --img_size 128 \
  --input_dir src/data/raw/images \
  --output_csv src/data/raw/predictions_calibrated.csv \
  --use_calibrated \
  --calibrator_path src/model/logit_calibrator.pkl \
  --uncertainty_threshold 0.25
```

Outputs:
- Predictions CSV with `TRUE` / `FALSE` / `?`
- `toloka_uncertain.csv` (only `?`) ready for Toloka markup

## 4) Evaluate

Evaluate predictions vs labels (optionally on a specific split reproduced with seed=42):
```bash
python src/model/eval_predictions.py \
  --predictions src/data/gt/pretrained_predictions.csv \
  --labels src/data/gt/labels.csv \
  --subset test \
  --seed 42 \
  --confusion_png src/data/gt/confusion_heatmap_test.png
```

Defaults: `subset=all`, `seed=42`. Metrics: accuracy, precision, recall, F1 (on confident predictions only), coverage, confusion counts, and a heatmap PNG.

## 5) Download data

Labeled images:
```bash
python main.py --type markup
```

Unlabeled images:
```bash
python main.py --type no_markup
```

## Notes
- Checkpoints are saved as `src/model/best_model_<model_type>_<model_name>.pth`.
- Training now auto-loads the checkpoint if present to continue fine-tuning.
- Calibration uses logits → logistic regression (Platt scaling).
- Uncertainty threshold 0.25 means probabilities in [0.25, 0.75] are marked as `?`.

