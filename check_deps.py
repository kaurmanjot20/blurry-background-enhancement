import sys
try:
    import cv2
    print("cv2: OK")
except ImportError:
    print("cv2: Missing")

try:
    import ultralytics
    print("ultralytics: OK")
except ImportError:
    print("ultralytics: Missing")

try:
    import yaml
    print("yaml: OK")
except ImportError:
    print("yaml: Missing (needed by ultralytics)")

try:
    import requests
    print("requests: OK")
except ImportError:
    print("requests: Missing (needed by ultralytics)")

try:
    from PIL import Image
    print("Pillow: OK")
except ImportError:
    print("Pillow: Missing (needed by ultralytics)")
