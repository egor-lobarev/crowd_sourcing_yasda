import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib

from pretrained_model import PretrainedTreeClassifier
from unet import UNet


class ImageDataset(Dataset):
    """Dataset for inference on unlabeled images"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            # Return a black image as fallback
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, str(img_path)


def get_transforms(img_size=128):
    """Get validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def predict_with_uncertainty(
    model,
    dataloader,
    device,
    uncertainty_threshold=0.25,
    calibrator=None,
    use_calibrated=False,
):
    """
    Predict on images and mark uncertain ones with '?'.
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: torch device
        uncertainty_threshold: Probability range for uncertainty (e.g., 0.25 means 0.25-0.75 is uncertain)
    
    Returns:
        List of dicts with filename, prediction, probability, logit
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = model(images)

            # Convert logits to numpy
            logits_np = logits.cpu().numpy().reshape(-1, 1)

            # Convert logits to probabilities
            if use_calibrated and calibrator is not None:
                probs = calibrator.predict_proba(logits_np)[:, 1]
            else:
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            for prob, logit_val, path in zip(probs, logits_np[:, 0], paths):
                filename = Path(path).name
                
                # Determine prediction based on probability and uncertainty
                if prob < uncertainty_threshold:
                    prediction = 'FALSE'
                elif prob > (1 - uncertainty_threshold):
                    prediction = 'TRUE'
                else:
                    # Uncertain - mark with '?'
                    prediction = '?'
                
                results.append({
                    'filename': filename,
                    'prediction': prediction,
                    'probability': float(prob),
                    'logit': float(logit_val),
                    'filepath': path
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict on unlabeled images and mark uncertain ones')
    parser.add_argument('--model_type', type=str, default='pretrained',
                       choices=['pretrained', 'unet'],
                       help='Model type: pretrained or unet')
    parser.add_argument('--model_name', type=str, default='convnext_tiny.fb_in22k',
                       help='Model name (for pretrained type)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint. If not specified, will use best_model_{model_type}_{model_name}.pth')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (should match training)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--input_dir', type=str, default='src/data/raw/images',
                       help='Directory with images to predict on')
    parser.add_argument('--output_csv', type=str, default='src/data/llm_markup/predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.25,
                       help='Uncertainty threshold: if prob is between [threshold, 1-threshold], mark as "?"')
    parser.add_argument('--use_calibrated', action='store_true',
                       help='Use calibrated logistic regression on logits (if available)')
    parser.add_argument('--calibrator_path', type=str, default='src/model/logit_calibrator.pkl',
                       help='Path to calibrated logistic regression model')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    # Model path
    if args.model_path is None:
        model_filename = f'best_model_{args.model_type}_{args.model_name if args.model_type == "pretrained" else "unet"}.pth'
        args.model_path = f'src/model/{model_filename}'
    
    print(f'Loading model from: {args.model_path}')
    
    # Load model
    if args.model_type == 'pretrained':
        model = PretrainedTreeClassifier(
            model_name=args.model_name,
            num_classes=2,
            img_size=args.img_size
        ).to(device)
    else:
        model = UNet().to(device)
    
    # Load checkpoint
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f'Error: Model checkpoint not found at {model_path}')
        print(f'Please train a model first or specify correct --model_path')
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model loaded successfully!')
    except Exception as e:
        print(f'Error loading model: {e}')
        return
    
    # Optionally load calibrator
    calibrator = None
    if args.use_calibrated:
        calib_path = Path(args.calibrator_path)
        if calib_path.exists():
            try:
                calibrator = joblib.load(calib_path)
                print(f'Loaded calibrated logistic regression from: {calib_path}')
            except Exception as e:
                print(f'Warning: could not load calibrator ({e}). Falling back to raw sigmoid.')
                calibrator = None
        else:
            print(f'Calibrator not found at {calib_path}, using raw sigmoid.')

    # Load images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'Error: Input directory {input_dir} not found!')
        return
    
    image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        print(f'No images found in {input_dir}')
        return
    
    print(f'Found {len(image_paths)} images')
    
    # Create dataset and dataloader
    transform = get_transforms(img_size=args.img_size)
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Predict
    print('Running predictions...')
    results = predict_with_uncertainty(
        model,
        dataloader,
        device,
        uncertainty_threshold=args.uncertainty_threshold,
        calibrator=calibrator,
        use_calibrated=args.use_calibrated,
    )
    
    # Save results
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df = df[['filename', 'prediction', 'probability', 'logit']]  # Reorder columns
    df.to_csv(output_path, index=False)
    
    # Print statistics
    total = len(results)
    true_count = sum(1 for r in results if r['prediction'] == 'TRUE')
    false_count = sum(1 for r in results if r['prediction'] == 'FALSE')
    uncertain_count = sum(1 for r in results if r['prediction'] == '?')
    
    print('\n' + '='*50)
    print('Prediction Summary:')
    print(f'Total images: {total}')
    print(f'TRUE (coniferous): {true_count} ({100*true_count/total:.1f}%)')
    print(f'FALSE (deciduous): {false_count} ({100*false_count/total:.1f}%)')
    print(f'? (uncertain): {uncertain_count} ({100*uncertain_count/total:.1f}%)')
    print(f'\nResults saved to: {output_path}')
    print('='*50)
    
    # Save separate file for Toloka (only uncertain images)
    if uncertain_count > 0:
        toloka_path = output_path.parent / 'toloka_uncertain.csv'
        uncertain_df = df[df['prediction'] == '?'].copy()
        uncertain_df.to_csv(toloka_path, index=False)
        print(f'\nUncertain images ({uncertain_count}) saved to: {toloka_path}')
        print('This file is ready for Toloka markup!')


if __name__ == '__main__':
    main()

