# Intelligent Background Enhancement Pipeline

This repository implements a lightweight, production-grade computer vision pipeline designed to selectively enhance background details in camera feeds while preserving foreground subjects. The system is engineered effectively for real-time applications using efficient neural/heuristic models.

## System Architecture

The pipeline follows a modular decision-based architecture:

```
Input Frame
    │
    ▼
[Person Detection (YOLO11n-seg)] ───► [No Subject?] ──┐
    │                                                 │
    ▼                                                 │
[Subject Mask Generation]                             │ (Skip Processing)
    │                                                 │
    ▼                                                 │
[Depth/Blur Estimation] ──────────────────────────────┘
    │
    ├─► If Subject Small: Skip (False Positive?)
    └─► If Background Blurry: Proceed
            │
            ▼
    [Background Extraction]
            │
            ▼
    [Restoration Module (MobileNet/OpenCV)]
    (Large-Scale Unsharp Masking + Detail Enhancement)
            │
            ▼
    [Alpha Blending] ◄── [Refined Mask (Feathered)]
            │
            ▼
       Final Output
```

## Core Modules

### 1. Detection & Segmentation (`camera_pipeline.detect`)
- **Model**: YOLO11n-seg (Nano Seg model).
- **Reasoning**: Chosen for its extreme efficiency (<3M params) and ability to run real-time on CPU.
- **Config**: Confidence threshold set to 0.5 to ensure high-precision subject targeting.

### 2. Decision Logic (`camera_pipeline.main`)
To maintain high throughput and artistic intent, the pipeline conditionally bypasses enhancement:
- **Presence Check**: If no person is detected, the image is passed through unaltered to avoid false sharpening of potential subjects.
- **Minimum Subject Size**: If `bbox_area / image_area < 0.01`, the subject is considered too far or a false detection. Enhancement is skipped.
- **Blur Variance**: Calculates Laplacian variance on the background. If the background is naturally sharp (Variance > 5000), processing is skipped to prevent artifacts.

### 3. Background Restoration (`camera_pipeline.deblur_cnn`)
The system employs a **Dual-Mode Restoration Engine** that bridges the gap between classical computer vision and deep learning:

#### Mode A: Heuristic (Default/Prototype)
Used when no trained weights are available.
- **Large-Scale Unsharp Masking**: Counters wide-radius defocus blur (${\sigma}=10.0$) to recover structure.
- **Detail Enhancement**: Synthesizes high-frequency texture using edge-preserving filters (`cv2.detailEnhance`).

#### Mode B: Neural (MobileNetDeblur)
Used when trained weights are loaded.
- **Architecture**: A lightweight Encoder-Decoder CNN inspired by MobileNetV2.
- **Performance**: Optimized for real-time CPU/Edge inference.
- **Training**: Can be trained on synthetic datasets to learn complex deblurring kernels.

## Training the AI Model

The repository now includes a full training suite to transition from Heuristic to Neural mode.

### 1. Generate Dataset
Create a synthetic dataset (Sharp/Blur pairs) from a folder of high-quality images:
```bash
python create_dataset.py --source /path/to/sharp_images --output dataset --count 1000
```
*Features*: Generates random motion blur and gaussian blur with noise to simulate real-world imperfections.

### 2. Train the Model
Train the `MobileNetDeblur` network using the generated dataset:
```bash
python train.py --data dataset --epochs 50 --batch_size 8
```
*Output*: Saves the best model to `best_model.pth`.

### 4. Blending Engine (`camera_pipeline.merge`)
- **Mask Refinement**: Raw segmentation masks are dilated (2 iterations) to cover hair/clothing fringes.
- **Feathering**: Gaussian blurred (5x5 kernel) to create a soft alpha channel.
- **Composition**: Linear alpha blending ensures no seamless artifacts or "halos" occur at the subject boundary.

## Setup & Usage

### Dependencies

We highly recommend using a virtual environment to avoid dependency conflicts (especially with PyTorch and Torchvision).

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*Requires: `opencv-python`, `ultralytics`, `torch`, `numpy`.*

### Running the Demo
The demo script handles webcam streams and static images (including 4-channel transparent PNGs), with auto-scaling for visualization.

**Webcam:**
```bash
python run_demo.py --source 0
```

**Static Image:**
```bash
python run_demo.py --source path/to/image.png
```
*(Note: Static images are processed instantly. For transparent PNGs, the alpha channel is automatically handled.)*

*Controls*: Press `s` to save the output, `q` to quit (for webcam streams).

## Engineering Trade-offs
- **Latency vs. Quality**: The pipeline prioritizes latency. Heavy generative models (GANs) were rejected in favor of efficient signal processing (Unsharp Mask/Detail Enhance) to enable CPU-only execution.
- **Safety First**: The "Do No Harm" policy is strictly enforced. If detection fails, enhancement is aborted to guarantee the subject is never accidentally distorted.
