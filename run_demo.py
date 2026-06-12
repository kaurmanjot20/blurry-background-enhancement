import cv2
import sys
import argparse
import numpy as np
import numpy as np
from camera_pipeline.main import EnhancementPipeline
from camera_pipeline.metrics import print_comparison_report

def stack_images(images, scale=0.5):
    """Stack images horizontally for visualization."""
    resized = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        resized.append(cv2.resize(img, new_size))
    return np.hstack(resized)

def main():
    parser = argparse.ArgumentParser(description="AI Camera Enhancement Demo")
    parser.add_argument("--source", type=str, default="0", help="Path to image or '0' for webcam")
    parser.add_argument("--scale", type=float, default=0.5, help="Visualization scale")
    args = parser.parse_args()

    pipeline = EnhancementPipeline(capture_mode=False)
    
    source = args.source
    if source.isdigit():
        source = int(source)
    
    is_image = False
    cap = None
    if isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        is_image = True
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return

    print("Press 'q' to quit. Press 's' to save current frame.")
    last_stacked = None
    last_frame = None
    last_output = None
    processed_image = False

    while True:
        if is_image:
            if processed_image:
                ret = False
            else:
                frame = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                ret = frame is not None
                processed_image = True
        else:
            ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            if not isinstance(source, int) and last_stacked is not None:
                # AUTOMATION: Auto-save and report for static images
                import os
                out_path = os.path.abspath("output_demo.jpg")
                cv2.imwrite(out_path, last_stacked)
                print(f"Saved result to: {out_path}")
                
                if last_frame is not None and last_output is not None:
                    print_comparison_report(last_frame, last_output)
            break
            
        print(f"Processing frame: {frame.shape}")

        # Convert 4-channel image (e.g., PNG with alpha) to 3-channel (BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize for speed if webcam is Huge
        if isinstance(source, int):
            # Optional: resize input to speed up detection and inference for demo
            pass
            
        # Process
        result = pipeline.process_frame(frame)
        print(f"Frame Status: {result['status']}")
        
        # Metrics to console
        m = result.get('metrics', {})
        total_dt = result['time_ms']
        fps = 1000.0 / total_dt if total_dt > 0 else 0
        print(f"  FPS: {fps:.1f} | Total: {total_dt:.1f}ms")
        print(f"  Det: {m.get('detect',0):.1f}ms | Chk: {m.get('check',0):.1f}ms | Enh: {m.get('enhance',0):.1f}ms | Bld: {m.get('blend',0):.1f}ms")
        
        # Visualize
        # Original | Mask | Enhanced BG | Final
        
        # Make mask visible
        mask_vis = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
        
        # Enhanced BG
        rec_bg = result['enhanced_bg']
        
        # Final
        final = result['output']
        
        # Cache for reporting
        last_frame = frame
        last_output = final
        
        # Calculate Auto-Scale to fit screen
        # We want to stack 4 images horizontally
        # Max total width target = 1600
        h, w = frame.shape[:2]
        total_w_orig = w * 4
        target_scale = 1600 / total_w_orig
        
        # Use simple reasonable max scale
        eff_scale = min(args.scale, target_scale)
        if eff_scale < 0.1: eff_scale = 0.1
        
        # Stack
        stacked = stack_images([frame, mask_vis, rec_bg, final], scale=eff_scale)
        last_stacked = stacked
        
        # Add Text
        cv2.putText(stacked, f"Status: {result['status']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Metrics
        m = result.get('metrics', {})
        total_dt = result['time_ms']
        fps = 1000.0 / total_dt if total_dt > 0 else 0
        
        cv2.putText(stacked, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(stacked, f"Total: {total_dt:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(stacked, f"Det: {m.get('detect',0):.1f}ms | Chk: {m.get('check',0):.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        cv2.putText(stacked, f"Enh: {m.get('enhance',0):.1f}ms | Bld: {m.get('blend',0):.1f}ms", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
                    
        cv2.imshow("Camera Enhancement Pipeline", stacked)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("output_demo.jpg", stacked)
            print("Saved output_demo.jpg")
            if frame is not None and final is not None:
                print_comparison_report(frame, final)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
