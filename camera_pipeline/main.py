import cv2
import numpy as np
import torch
import time
from .detect import PersonDetector
from .blur_detection import estimate_blur
from .segment import refine_mask
from .detect import PersonDetector
from .deblur_cnn import MobileNetDeblur
from .merge import blend_images

class EnhancementPipeline:
    def __init__(self, use_cuda=True, capture_mode=False):
        """
        Initialize the AI pipeline.
        
        Args:
            use_cuda (bool): Whether to use GPU.
            capture_mode (bool): If True, use higher quality settings (slower).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        print(f"Pipeline running on: {self.device}")
        
        # Initialize modules
        self.detector = PersonDetector(conf_thresh=0.5)
        self.model = MobileNetDeblur(pretrained=False).to(self.device)
        self.model.eval()
        
        # Parameters
        self.capture_mode = capture_mode
        self.close_thresh = 0.01 # Relaxed: process almost any person size
        self.blur_thresh = 5000.0 # Relaxed: process almost anything (unless extremely sharp)
        
    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame.
        
        Returns:
            dict: {
                'output': np.ndarray,
                'mask': np.ndarray,
                'enhanced_bg': np.ndarray,
                'status': str,
                'time_ms': float
            }
        """

        t0 = time.time()
        metrics = {'detect': 0.0, 'check': 0.0, 'enhance': 0.0, 'blend': 0.0}
        
        # 1. Detection
        det_res = self.detector.detect(frame)
        t1 = time.time()
        metrics['detect'] = (t1 - t0) * 1000
        
        
        if not det_res['has_subject']:
            return self._pack_result(frame, det_res, "No subject detected (Skipped)", t0, metrics=metrics)

        # 2. Decision Logic
        # Check closeness
        if det_res['bbox_area_ratio'] < self.close_thresh:
            t_check = (time.time() - t1) * 1000
            metrics['check'] = t_check
            return self._pack_result(frame, det_res, "Subject too far", t0, metrics=metrics)
            
        # Check background blur
        # Invert mask to get background
        bg_mask = cv2.bitwise_not(det_res['mask'])
        blur_score = estimate_blur(frame, bg_mask)
        
        t2 = time.time()
        metrics['check'] = (t2 - t1) * 1000
        
        if blur_score > self.blur_thresh:
             return self._pack_result(frame, det_res, f"Background sharp ({blur_score:.1f})", t0, metrics=metrics)
             
        # 3. Enhance Background
        # Prepare input for CNN
        # Resize if necessary (lightweight model might like smaller input)
        # For prototype, we pass full res or downscale
        
        # Preprocess
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = self.model(img_tensor)
            
        # Postprocess
        enhanced_bg = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        enhanced_bg = np.clip(enhanced_bg, 0, 255).astype(np.uint8)
        
        t3 = time.time()
        metrics['enhance'] = (t3 - t2) * 1000
        
        # 4. Refine Mask & Blend
        # We need to refine the subject mask
        mask_refined = refine_mask(det_res['mask'], dilate_iter=2, feather_amount=5)
        
        final = blend_images(frame, enhanced_bg, mask_refined)
        
        t4 = time.time()
        metrics['blend'] = (t4 - t3) * 1000
        
        return self._pack_result(final, det_res, f"Enhanced (Blur: {blur_score:.1f})", t0, enhanced_bg=enhanced_bg, metrics=metrics)

    def _pack_result(self, output, det_res, status, start_time, enhanced_bg=None, metrics=None):
        dt = (time.time() - start_time) * 1000
        # If enhanced_bg is None (skipped), show original so viz isn't black
        viz_bg = enhanced_bg if enhanced_bg is not None else output.copy()
        
        if metrics is None:
            metrics = {}
            
        return {
            'output': output,
            'mask': det_res['mask'],
            'enhanced_bg': viz_bg,
            'status': status,
            'time_ms': dt,
            'metrics': metrics
        }
