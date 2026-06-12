import cv2
import numpy as np

def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    """
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_complex_metrics(image):
    """
    Calculate extensive reference-less metrics.
    """
    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    metrics = {}
    
    # --- Traditional Sharpness ---
    
    # 1. Laplacian Variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['laplacian_var'] = lap.var()
    
    # 2. Tenengrad
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = gx**2 + gy**2
    metrics['tenengrad'] = np.mean(mag)
    
    # 3. Brenner Gradient
    # difference between pixel and neighbor 2 steps away
    diff_x = np.subtract(gray[:, 2:], gray[:, :-2], dtype=np.float64)
    diff_y = np.subtract(gray[2:, :], gray[:-2, :], dtype=np.float64)
    metrics['brenner'] = np.mean(diff_x**2) + np.mean(diff_y**2)
    
    # 4. Modified Laplacian
    # Sum of absolute laplacian
    lx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    ly = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
    # Approx: |2*I - I(x-1) - I(x+1)| + ... 
    # Use standard laplacian absolute
    lap_abs = np.abs(lap)
    metrics['mod_laplacian'] = np.mean(lap_abs)
    
    # 5. High Frequency Content (HFC)
    # Usually sum of transform coeffs, but here we estimate via gradient magnitude relative to brightness
    metrics['high_freq'] = np.mean(mag) / (np.mean(gray) + 1e-6)

    # 6. Edge Density
    edges = cv2.Canny(gray, 100, 200)
    metrics['edge_density'] = np.count_nonzero(edges) / edges.size

    # --- Perceptual ---
    
    # 7. Local Contrast
    # Standard deviation of local blocks. 
    # Compute std dev in 7x7 blocks.
    # Efficient: Blur(I^2) - Blur(I)^2
    mu = cv2.blur(gray.astype(float), (7,7))
    sq = cv2.blur(gray.astype(float)**2, (7,7))
    sigma = np.sqrt(np.maximum(sq - mu**2, 0))
    metrics['local_contrast'] = np.mean(sigma)
    
    # 8. Acutance
    # Gradient Mean / Intensity Mean
    metrics['acutance'] = np.mean(np.sqrt(mag)) / (np.mean(gray) + 1e-6)

    return metrics

def print_comparison_report(original, enhanced):
    """
    Prints a formatted comparison table between two images.
    """
    m_orig = calculate_complex_metrics(original)
    m_enh = calculate_complex_metrics(enhanced)
    
    # Structural Metrics (Compare pairs)
    psnr = cv2.PSNR(original, enhanced)
    ssim = calculate_ssim(original, enhanced)
    
    print("\n" + "="*80)
    print(f"{'COMPREHENSIVE DEBLURRING PERFORMANCE METRICS':^80}")
    print("="*80)
    
    print("\n[A] TRADITIONAL SHARPNESS METRICS:")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Original':<15} | {'Enhanced':<15} | {'Change':<10}")
    print("-" * 80)
    
    order = ['laplacian_var', 'tenengrad', 'brenner', 'mod_laplacian', 'high_freq', 'edge_density']
    for key in order:
        v1 = m_orig.get(key, 0)
        v2 = m_enh.get(key, 0)
        
        if v1 == 0: change = 0.0
        else: change = ((v2 - v1) / v1) * 100
        
        sign = "+" if change > 0 else ""
        name = key.replace("_", " ").title()
        
        print(f"{name:<25} | {v1:<15.2f} | {v2:<15.2f} | {sign}{change:.1f}%")

    print("\n[B] PERCEPTUAL QUALITY METRICS:")
    print("-" * 80)
    for key in ['local_contrast', 'acutance']:
        v1 = m_orig.get(key, 0)
        v2 = m_enh.get(key, 0)
        
        if v1 == 0: change = 0.0
        else: change = ((v2 - v1) / v1) * 100
        sign = "+" if change > 0 else ""
        name = key.replace("_", " ").title()
        print(f"{name:<25} | {v1:<15.2f} | {v2:<15.2f} | {sign}{change:.1f}%")

    print("\n[C] STRUCTURAL SIMILARITY (Input vs Output):")
    print("-" * 80)
    print(f"{'PSNR (dB)':<25}: {psnr:.2f} dB")
    print(f"{'SSIM Index':<25}: {ssim:.4f}")
    print("Note: Since we lack a perfect ground truth, these measure fidelity" )
    print("      to the original input. Lower is often better for enhancement" )
    print("      if the original was very blurry.")
    
    print("="*80 + "\n")
