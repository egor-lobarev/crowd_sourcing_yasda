# crowd_sourcing_yasda
The repository of group markup using toloka and LLM and predicting.

## Usage model training:

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