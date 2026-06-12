import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolo11n-seg.pt", conf_thresh=0.25):
        """
        Initialize the YOLOv11 detector.
        """
        print(f"Loading YOLO model: {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading yolo11n-seg.pt...")
            self.model = YOLO("yolo11n-seg.pt")
            
        self.conf_thresh = conf_thresh

    def detect(self, image: np.ndarray):
        """
        Detect subjects (persons, objects, etc.) in the image.
        
        Args:
            image (np.ndarray): Input BGR image.
        
        Returns:
            dict: {
                'has_subject': bool,
                'bbox_area_ratio': float,
                'mask': np.ndarray (uint8, 0-255, same size as image),
                'viz_plot': np.ndarray (optional visualization from YOLO)
            }
        """
        h, w = image.shape[:2]
        
        # Run inference (detect ALL classes)
        results = self.model(image, conf=self.conf_thresh, verbose=False)
        result = results[0]
        
        output = {
            'has_subject': False,
            'bbox_area_ratio': 0.0,
            'mask': np.zeros((h, w), dtype=np.uint8),
            'viz_plot': result.plot()
        }
        
        if result.boxes is None or len(result.boxes) == 0:
            return output
            
        output['has_subject'] = True
        
        # Calculate max area ratio of the largest subject
        max_ratio = 0.0
        
        # Combine all masks into one
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            # Masks are usually lower resolution, resize them
            # masks.data is a torch tensor (N, H_mask, W_mask)
            masks_data = result.masks.data.cpu().numpy()
            
            # The mask provided by YOLO is relative to the input size passed to the model,
            # but usually Ultralytics handles aspect ratio. 
            # We need to resize strictly to (w, h).
            # Note: masks_data values are float or bool? Usually float 0..1 or bool.
            
            for i, mask_tensor in enumerate(masks_data):
                # Resize to original image size
                # mask_tensor is (mh, mw)
                m = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_LINEAR)
                m = (m > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, m)
                
                # BBox area for this subject
                box = result.boxes[i]
                # xywh = box.xywh[0].cpu().numpy() # center_x, center_y, w, h
                # area = xywh[2] * xywh[3]
                
                # Better: use the mask area or box area? 
                # Prompt says: "bounding_box_area / image_area"
                if len(box.xywhn) > 0:
                    # xywhn is normalized 0-1
                    bw = box.xywhn[0][2].item()
                    bh = box.xywhn[0][3].item()
                    ratio = bw * bh
                    if ratio > max_ratio:
                        max_ratio = ratio

        output['mask'] = combined_mask
        output['bbox_area_ratio'] = max_ratio
        
        return output
