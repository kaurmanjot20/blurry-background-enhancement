import cv2
import numpy as np

def refine_mask(mask: np.ndarray, dilate_iter: int = 2, feather_amount: int = 5) -> np.ndarray:
    """
    Refine a binary segmentation mask to prepare for seamless blending.
    
    Steps:
    1. Dilation: Expands the foreground slightly to cover potential segmentation errors at edges.
    2. Feathering: Blurs the edge to create a soft transition (alpha channel).
    
    Args:
        mask (np.ndarray): Binary mask (0 or 255/1), single channel.
        dilate_iter (int): Number of dilation iterations.
        feather_amount (int): Gaussian blur kernel size for feathering (must be odd).
    
    Returns:
        np.ndarray: Float32 mask in range [0, 1] for alpha blending.
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # 1. Dilation
    # Use a circular kernel for smoother expansion
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # We dilate the PERSON mask (foreground). 
    # If the input is the person mask, we dilate it to ensure we don't sharpen the fringe of the person.
    refined = cv2.dilate(mask, kernel, iterations=dilate_iter)
    
    # 2. Feathering (Gaussian Blur)
    # Ensure feather_amount is odd
    if feather_amount % 2 == 0:
        feather_amount += 1
    
    # Apply blur
    refined_float = refined.astype(np.float32) / 255.0
    refined_blurred = cv2.GaussianBlur(refined_float, (feather_amount, feather_amount), 0)
    
    # Clip to be safe
    return np.clip(refined_blurred, 0.0, 1.0)

def extract_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract the background region. 
    Strict extraction isn't always needed if we mask the output, 
    but this helps if the model needs black foreground.
    
    Args:
        image (np.ndarray): Original image.
        mask (np.ndarray): Binary mask of Foreground (Person).
    
    Returns:
        np.ndarray: Image with foreground blacked out (optional approach) 
                    or just returns the image if the model handles full image.
                    Here we return the full image because the CNN might use context,
                    but the training usually dictates this. 
                    For inference on background only, we usually pass the whole image 
                    and mix later, OR we replace foreground with mean color to avoid artifacts.
                    Let's just return the image as is for the pipeline logic, 
                    the spatial masking happens at the blending stage.
    """
    return image
