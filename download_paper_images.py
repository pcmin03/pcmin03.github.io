#!/usr/bin/env python3
"""
Script to download images from ar5iv HTML versions of arXiv papers
"""
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse
import sys

def download_images_from_ar5iv(arxiv_id, output_dir):
    """
    Download all images from an ar5iv HTML page
    """
    ar5iv_url = f"https://ar5iv.org/abs/{arxiv_id}"
    
    print(f"Fetching {ar5iv_url}...")
    try:
        response = requests.get(ar5iv_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {ar5iv_url}: {e}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all images
    images = soup.find_all('img')
    
    downloaded_files = []
    
    for idx, img in enumerate(images, 1):
        img_src = img.get('src') or img.get('data-src')
        if not img_src:
            continue
        
        # Skip small icons/logos and base64 images
        if 'logo' in img_src.lower() or 'icon' in img_src.lower() or img_src.startswith('data:'):
            continue
        
        # Make absolute URL
        if img_src.startswith('//'):
            img_url = 'https:' + img_src
        elif img_src.startswith('/'):
            img_url = 'https://ar5iv.org' + img_src
        elif not img_src.startswith('http'):
            img_url = urljoin(ar5iv_url, img_src)
        else:
            img_url = img_src
        
        # Get filename
        parsed = urlparse(img_url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            ext = img_src.split('.')[-1].split('?')[0]
            if ext in ['png', 'jpg', 'jpeg', 'gif', 'svg']:
                filename = f"figure{idx}.{ext}"
            else:
                filename = f"figure{idx}.png"
        
        # Clean filename
        filename = re.sub(r'[^\w\.-]', '_', filename)
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"  Downloading {filename}...")
            img_response = requests.get(img_url, timeout=30)
            img_response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            
            downloaded_files.append(filepath)
            print(f"    Saved to {filepath}")
        except Exception as e:
            print(f"    Error downloading {img_url}: {e}")
    
    return downloaded_files

def main():
    # Paper mappings: arxiv_id -> output_directory
    # Add arXiv IDs here as you find them
    papers = {
        '2302.06081': 'correspondence-free-domain-alignment',  # Correspondence-free domain alignment - DONE
        '2405.20645': 'semantic-feature-learning',  # Semantic feature learning universal retrieval (NeurIPS 2024)
        # TODO: Find arXiv IDs for:
        # - Domain-generalized cross-domain retrieval (ICCV 2023) - may not be on arXiv
        # - Prototypical optimal transport (AAAI 2024) - need to find
        # - Noise mitigation (ICME 2025) - may not be on arXiv yet
    }
    
    base_dir = 'assets/images/posts'
    
    # Also try to download from ar5iv HTML pages directly
    # ar5iv URL format: https://ar5iv.org/abs/2302.06081 or https://ar5iv.labs.arxiv.org/html/2302.06081
    
    for arxiv_id, dir_name in papers.items():
        output_dir = os.path.join(base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing paper {arxiv_id} -> {dir_name}")
        downloaded = download_images_from_ar5iv(arxiv_id, output_dir)
        print(f"Downloaded {len(downloaded)} images")
        
        # Also try the labs.arxiv.org format
        if len(downloaded) <= 1:  # Only logo downloaded
            print(f"Trying alternative ar5iv format...")
            try:
                alt_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
                response = requests.get(alt_url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    images = soup.find_all('img')
                    for idx, img in enumerate(images, 1):
                        img_src = img.get('src') or img.get('data-src')
                        if not img_src or 'logo' in img_src.lower():
                            continue
                        if img_src.startswith('//'):
                            img_url = 'https:' + img_src
                        elif img_src.startswith('/'):
                            img_url = 'https://ar5iv.labs.arxiv.org' + img_src
                        elif not img_src.startswith('http'):
                            img_url = urljoin(alt_url, img_src)
                        else:
                            img_url = img_src
                        
                        filename = f"figure{idx}.png"
                        filepath = os.path.join(output_dir, filename)
                        try:
                            img_response = requests.get(img_url, timeout=30)
                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)
                            print(f"  Downloaded {filename}")
                            downloaded.append(filepath)
                        except:
                            pass
            except Exception as e:
                print(f"  Alternative format failed: {e}")

if __name__ == '__main__':
    main()

