import cv2
import numpy as np

def blend_images(original: np.ndarray, enhanced_bg: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """
    Blend the original foreground with the enhanced background using the alpha mask.
    
    Args:
        original (np.ndarray): Original image (contains sharpness of person).
        enhanced_bg (np.ndarray): Image with sharpened background.
        alpha_mask (np.ndarray): Float32 mask [0, 1]. 
                                 1.0 = Foreground (Person), 0.0 = Background.
    
    Returns:
        np.ndarray: Final blended image.
    """
    # Ensure types match
    if original.dtype != np.float32:
        orig_f = original.astype(np.float32)
    else:
        orig_f = original
        
    if enhanced_bg.dtype != np.float32:
        enh_f = enhanced_bg.astype(np.float32)
    else:
        enh_f = enhanced_bg
        
    # Ensure mask dimensions
    if len(alpha_mask.shape) == 2:
        alpha = alpha_mask[:, :, np.newaxis]
    else:
        alpha = alpha_mask
        
    # Blend:
    # Output = Original * Alpha + EnhancedBG * (1 - Alpha)
    # Because Alpha=1 is the Person (Original)
    
    out = orig_f * alpha + enh_f * (1.0 - alpha)
    
    return np.clip(out, 0, 255).astype(np.uint8)
