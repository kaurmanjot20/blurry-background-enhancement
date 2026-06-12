import cv2
import numpy as np

def estimate_blur(image: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Estimate the blurriness of an image or a specific region using the 
    variance of the Laplacian method.
    
    Args:
        image (np.ndarray): BGR or Grayscale image.
        mask (np.ndarray): Optional binary mask (uint8) where 255 indicates the region of interest.
                           If None, the entire image is used.
    
    Returns:
        float: The variance of the Laplacian. Lower values indicate more blur.
    """
    if image is None:
        return 0.0

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None:
        # Ensure mask is same size
        if mask.shape[:2] != gray.shape[:2]:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Select pixels in the region
        # We only care about the variance of the laplacian values in the background
        # To avoid edge effects at the mask boundary, we might want to erode the mask slightly if generic,
        # but inputs should be handled by caller.
        
        # Extract values where mask is present (background)
        # Assuming mask is 255 for ROI
        values = laplacian[mask > 127]
        
        if len(values) == 0:
            return 0.0
            
        return values.var()
    else:
        return laplacian.var()
