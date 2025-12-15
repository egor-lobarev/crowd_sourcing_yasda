# crowd_sourcing_yasda
The repository of group markup using toloka and LLM and predicting.

## Usage

### Model Training:

**Train with pretrained model (recommended):**
```bash
python src/model/train.py --model pretrained --model_name swin_tiny_patch4_window7_224 --img_size 128 --batch_size 16 --epochs 50
```

**Train with ConvNeXt-Tiny:**
```bash
python src/model/train.py --model pretrained --model_name convnext_tiny.fb_in22k --img_size 128
```

**Train with UNet (original):**
```bash
python src/model/train.py --model unet --img_size 256
```

### Inference on Unlabeled Images:

**Predict on images and mark uncertain ones with '?':**
```bash
python src/model/inference.py --model_type pretrained --model_name convnext_tiny.fb_in22k --img_size 128
```

**With custom uncertainty threshold (default: 0.25, meaning 0.25-0.75 probability range is uncertain):**
```bash
python src/model/inference.py --model_type pretrained --model_name convnext_tiny.fb_in22k --uncertainty_threshold 0.3
```

The script will:
- Process all images from `src/data/raw/images/`
- Mark predictions as: `TRUE` (coniferous), `FALSE` (deciduous), or `?` (uncertain)
- Save results to `src/data/llm_markup/predictions.csv`
- Create `src/data/llm_markup/toloka_uncertain.csv` with only uncertain images for Toloka markup

### Download Images:

**Download labeled images:**
```bash
python main.py --type markup
```

**Download unlabeled images:**
```bash
python main.py --type no_markup
```