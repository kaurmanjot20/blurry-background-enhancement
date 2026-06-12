import cv2
import numpy as np
from camera_pipeline.metrics import print_comparison_report

def run():
    img_path = "output_demo.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: content not found.")
        return

    h, w = img.shape[:2]
    # Assuming 4 horizontal panels
    panel_w = w // 4
    
    original = img[:, 0:panel_w]
    final = img[:, panel_w*3:panel_w*4]
    
    import contextlib
    
    print(f"Extracted panels from {img_path} (WxH: {w}x{h})")
    print(f"Panel Width: {panel_w}")
    
    with open("metrics_report.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            print_comparison_report(original, final)
            
    print("Report saved to metrics_report.txt")

if __name__ == "__main__":
    run()
