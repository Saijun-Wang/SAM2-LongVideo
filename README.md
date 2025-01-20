# SAM2-LongVideo: a robust tracking system based on SAM2 and YOLO

## Overview

SAM2-LongVideo is a robust method for Video Object Segmentation (VOS) that combines the Segment Anything Model 2 (SAM2) with the YOLOv11. This approach focuses on two major challenges in long video segmentation: object tracking loss and high memory consumption. By dividing long video sequences into smaller sub-sequences, SAM2-LongVideo optimizes memory usage and enhances segmentation consistency.

This repository provides a solution for real-time, zero-shot segmentation of long video sequences, leveraging SAM2’s segmentation power with YOLO’s fast detection capabilities.

## Features

- **Zero-shot segmentation**: Leverages SAM2 for segmenting any object without prior training.
- **Efficient memory usage**: Splits long videos into smaller sub-sequences to reduce memory requirements.
- **Object tracking consistency**: Combines YOLOv11 for real-time bounding box prompt generation and SAM2 for segmentation.
- **Parallelizable processing**: Segments each sub-sequence independently, making the process parallelizable.
- **Suitable for resource-constrained environments**: Optimized for use in systems with limited GPU memory.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Saijun-Wang/SAM2-LongVideo.git
   cd SAM2-LongVideo
   ```
Then open anaconda prompt, and create our virtual envioronment.
   ```bash
   conda create -n SAM2_test python=3.11
   ```

2. Install required packages:
   With the cuda 11.8
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
   pip install hydra-core
   pip install tqdm
   pip install matplotlib
   pip install opencv-python
   pip install ninja
   pip install imageio
   pip install ultralytics
   ```
Then use the installation command, make sure that the the working directory is in the segment-anything-2 folder
   ```bash
   python setup.py build_ext --inplace
   ```

3. Download the SAM2 large model and put the .pt file in the "...\segment-anything-2\checkpoints" folder.
   ```bash
   https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
   ```
   
## Usage

### 1. Prepare Your Video

Convert your video into frames, and it would be better if each frame looks different.

### 2. Run Segmentation

Once the frames are ready, you can modify the parameters in the `config.yaml` and start segmentation using the `SAM2-LongVideo.py` script. It can track one object in a video.
There are two videos as examples in the `./notebooks/videos` folder, along with two corresponding pre-trained `.pt` files.
If you want to track an object in a video, you can directly use `SAM2-LongVideo.py`. For the example video of two mice, if you want to track both mice simultaneously, you can use the provided `track2mice.py`.

### 3. View Results

Segmentation results will be saved in the specified output directory.

Here’s how you can describe your parameters in the `README.md` file:

---

## Parameters

Below are the parameters used in the code for video object segmentation and YOLO training:

- **`use_pretrained_YOLO`**: `True`  
  If set to `True`, the pretrained YOLO model will be used for detection.
  If set to `False`, the method will use the first subsequence to train YOLO.

- **`yolo_model`**: `"best.pt"`  
  Path to the pretrained YOLO model file.
  If `use_pretrained_YOLO` is `True`, use your .pt document.
  If `use_pretrained_YOLO` is `False`, `yolo11s.pt` is recommened.

- **`sam2_checkpoint`**: `"checkpoints/sam2_hiera_large.pt"`  
  Path to the SAM2 checkpoint file for segmentation tasks.

- **`sam2_cfg`**: `"sam2_hiera_l.yaml"`  
  Configuration file for SAM2.

- **`train_number`**: `250`  
  The number of frames used for training the YOLO model. The first `train_number` frames are selected for training.

- **`train_epoch`**: `300`  
  The number of training epochs for YOLO.

- **`train_patience`**: `50`  
  The patience for early stopping during YOLO training.

- **`yolo_target_size`**: `640`  
  When computing resources are limited, you can resize the images to a smaller size before training YOLO to reduce memory usage.

- **`save_sam2_results`**: `True`  
  If set to `True`, the segmentation results from SAM2 will be saved.

- **`save_yolo_results`**: `True`  
  If set to `True`, the detection results from YOLO will be saved.

- **`video_dir`**: `"notebooks/videos/dog"`  
  Directory containing the video files to be processed.

- **`object_name`**: `"dog"`  
  The name of the object to be tracked and segmented in the video.

- **`output_dir`**: `"sam2_output_segment"`  
  Directory to save SAM2 segmentation results.

- **`output_yolo_dir`**: `"yolo_output_bbox"`  
  Directory to save YOLO detection results (bounding boxes).

- **`propagate_folder_size`**: `100`  
  The number of frames to process in each sub-sequence for memory optimization.

- **`subsequence`**: `1`  
  The folder number to start processing.

- **`confidence_threshold`**: `0.9`  
  Threshold for YOLO detection confidence. Only detections with confidence greater than this value will be considered as valid prompts.

- **`click`**: `False`  
  Set to `True` if you want to manually click on the first image to provide a point prompt.
  Set to `False` if you prefer to provide coordinates and labels for the prompt.

- **`coordinates`**: `[[701, 446], [245, 176]]`  
  The coordinates for the points where you want to initiate the segmentation (only used when `click=False`).

- **`labels`**: `[1, 0]`  
  The labels corresponding to each point, used when providing coordinates for the prompt.
