import os
import requests
import pandas as pd
from pathlib import Path

def download_images():
    """Download images from URLs in hw_3_markup_data.txt and save to src/data/gt/images/"""
    data_dir = Path('src/data/gt')
    images_dir = data_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    markup_file = data_dir / 'hw_3_markup_data.txt'
    
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

if __name__ == '__main__':
    download_images()