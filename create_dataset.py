import cv2
import numpy as np
import os
import random
import argparse
from glob import glob
from tqdm import tqdm

def apply_motion_blur(image, kernel_size=15):
    """
    Simulate motion blur with a random angle.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Random angle
    angle = random.randint(0, 180)
    
    # Create the kernel
    center = kernel_size // 2
    matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    
    # Draw a line on the kernel to simulate motion
    kernel[center, :] = 1
    # Rotate
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    # Apply
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_gaussian_blur(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Deblur Dataset")
    parser.add_argument("--source", type=str, required=True, help="Folder containing sharp images")
    parser.add_argument("--output", type=str, default="dataset", help="Output folder")
    parser.add_argument("--count", type=int, default=0, help="Max images to process (0 = all)")
    args = parser.parse_args()
    
    sharp_dir = os.path.join(args.output, "sharp")
    blur_dir = os.path.join(args.output, "blur")
    
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)
    
    # Find images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(args.source, ext)))
        # Also check uppercase
        files.extend(glob(os.path.join(args.source, ext.upper())))
        
    print(f"Found {len(files)} unique images in {args.source}")
    
    if args.count > 0:
        files = files[:args.count]
        
    for fpath in tqdm(files):
        try:
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            # Resize if too huge (to save space and time)
            h, w = img.shape[:2]
            if max(h, w) > 1920:
                scale = 1920 / max(h, w)
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
            # Save Sharp
            fname = os.path.basename(fpath)
            cv2.imwrite(os.path.join(sharp_dir, fname), img)
            
            # Generate Blur
            # Randomly choose blur type or mix
            r = random.random()
            if r < 0.4:
                # Motion Blur
                k = random.randrange(5, 30, 2) # Odd numbers
                blurred = apply_motion_blur(img, kernel_size=k)
            elif r < 0.8:
                # Gaussian Blur
                k = random.randrange(5, 25, 2)
                blurred = apply_gaussian_blur(img, kernel_size=k)
            else:
                # Mix
                blurred = apply_motion_blur(img, kernel_size=9)
                blurred = apply_gaussian_blur(blurred, kernel_size=5)
                
            # Add some noise for realism
            noise = np.random.normal(0, 5, blurred.shape).astype(np.uint8)
            blurred = cv2.add(blurred, noise)
            
            cv2.imwrite(os.path.join(blur_dir, fname), blurred)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    print(f"Dataset generated at: {args.output}")
    print(f"Sharp images: {sharp_dir}")
    print(f"Blur images:  {blur_dir}")

if __name__ == "__main__":
    main()
