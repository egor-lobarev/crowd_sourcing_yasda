import requests
import pandas as pd
from pathlib import Path
import argparse

def download_images():
    """Download images from URLs in hw_3_markup_data.txt and save to src/data/gt/images/"""
    data_dir = Path('src/data/test')
    images_dir = data_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    markup_file = data_dir / 'model_check.csv'
    
    if not markup_file.exists():
        print(f'Error: {markup_file} not found!')
        return
    
    with open(markup_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    data = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print(f'Skipping malformed line {i+1}: {line.strip()}')
            continue
        
        label_str, url = parts
        label = 1 if label_str == 'TRUE' else 0
        filename = f'{i:03d}.jpg'
        filepath = images_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            print(f'Skipping {filename} (already exists)')
            data.append({'filename': filename, 'label': label})
            continue
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, 'wb') as img_file:
                img_file.write(response.content)
            data.append({'filename': filename, 'label': label})
            print(f'Downloaded {filename} ({i+1}/{len(lines)})')
        except Exception as e:
            print(f'Failed to download {url}: {e}')
    
    # Save labels CSV
    df = pd.DataFrame(data)
    labels_file = data_dir / 'labels.csv'
    df.to_csv(labels_file, index=False)
    print(f'\nDownload complete. {len(data)} images processed.')
    print(f'Labels saved to {labels_file}')

def download_no_markup_images():
    """Download images from URLs in hw_3_no_markup_data.txt and save to src/data/raw/images/"""
    data_dir = Path('src/data/test')
    images_dir = data_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    no_markup_file = data_dir / 'model_check.csv'
    
    if not no_markup_file.exists():
        print(f'Error: {no_markup_file} not found!')
        return
    
    with open(no_markup_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        url = line.strip()
        if not url.startswith('http'):
            print(f'Skipping invalid line {i+1}: {url[:50]}...')
            failed += 1
            continue
        
        # Extract filename from URL or use index
        try:
            filename = url.split('/')[-1]
            if not filename.endswith('.jpg'):
                filename = f'{i:06d}.jpg'
        except:
            filename = f'{i:06d}.jpg'
        
        filepath = images_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f'Progress: {i+1}/{len(lines)} (skipped: {skipped}, downloaded: {downloaded}, failed: {failed})')
            continue
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, 'wb') as img_file:
                img_file.write(response.content)
            downloaded += 1
            
            if (i + 1) % 100 == 0 or downloaded % 100 == 0:
                print(f'Progress: {i+1}/{len(lines)} (downloaded: {downloaded}, skipped: {skipped}, failed: {failed})')
        except Exception as e:
            failed += 1
            if failed <= 10:  # Only print first 10 failures to avoid spam
                print(f'Failed to download {url[:50]}...: {e}')
            elif failed == 11:
                print('... (suppressing further error messages)')
    
    print('\n' + '='*50)
    print('Download complete!')
    print(f'Total URLs: {len(lines)}')
    print(f'Downloaded: {downloaded}')
    print(f'Skipped (already exists): {skipped}')
    print(f'Failed: {failed}')
    print(f'Images saved to: {images_dir}')
    print(f'{"="*50}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from markup or no-markup data files')
    parser.add_argument('--type', type=str, default='markup', 
                       choices=['markup', 'no_markup'],
                       help='Type of data to download: markup (with labels) or no_markup (unlabeled)')
    
    args = parser.parse_args()
    
    if args.type == 'markup':
        download_images()
    else:
        download_no_markup_images()