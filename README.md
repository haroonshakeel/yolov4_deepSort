# yolov4-deepsort
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This repository is created for the Youtube channel [TheCodingBug](https://www.youtube.com/channel/UCcNgapXcZkyW10FIOohZ1uA?sub_confirmation=1).

This repository shows how to use YOLOv4 and YOLOv4-Tiny to track objects using ``Deep Sort``. This is tested with ``TensorFlow 2.3.1`` on ``GTX-1060``.


# Object Tracking Results Yolov4 original weights

### Tracking ``person``
<p align="center"><img src="person.gif" width="640"\></p>

### Tracking ``dog``
<p align="center"><img src="dog.gif" width="640"\></p>

### Tracking ``dog`` and ``person``
<p align="center"><img src="dog_person.gif" width="640"\></p>

# Table of Contents

1. [Setting Up Environment](#setting-up-environment)
   * [Using Conda](#using-conda)
   * [Using Pip](#using-pip)
2. [Download Weights File](#download-eights-file)
3. [Convert YOLOv4 to TensorFlow](#convert-yolov4-to-tensorflow)
4. [Run Tracking](#run-tracking)
5. [Credits](#credits)



# Setting Up Environment
### Using Conda
#### CPU
```bash
# CPU
conda env create -f conda-cpu.yml


# activate environment on Windows or Linux
conda activate tf-cpu

# activate environment on Mac
source activate tf-cpu
```
#### GPU
```bash
# GPU
conda env create -f conda-gpu.yml

# activate environment on Windows or Linux
conda activate tf-gpu

# activate environment on Mac
source activate tf-gpu
```

### Using Pip
```bash
# CPU
pip install -r requirements.txt

# GPU
pip install -r requirements-gpu.txt

```
**Note:** If installing GPU version with Pip, you need to install CUDA and cuDNN in your system. You can find the tutorial for Windows [here](https://www.youtube.com/watch?v=PlW9zAg4cx8).

# Download Weights File
Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )

If using ``tiny`` version, download [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) file instead. ``tiny`` version is faster, but less accurate.

# Convert YOLOv4 to TensorFlow

```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

## yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

```
If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` in command and also download corresponding YOLOv3 weights and and change ``--weights to ./data/yolov3.weights``

### Run Tracking

```bash
# Run Tracking on Video
python track_objects.py --weights ./checkpoints/yolov4-416 --score 0.3 --video ./data/dog.mp4 --output ./results/demo.avi --model yolov4

# Run Tracking on Webcam
python track_objects.py --weights ./checkpoints/yolov4-416 --score 0.3 --video 0 --output ./results/webcam.avi --model yolov4

# Run Tracking on Video With Tiny Yolov4
python track_objects.py --weights ./checkpoints/yolov4-tiny-416 --score 0.3 --video ./data/dog.mp4 --output ./results/demo_tiny.avi --model yolov4

# Run Tracking on Webcam With Tiny Yolov4
python track_objects.py --weights ./checkpoints/yolov4-tiny-416 --score 0.3 --video 0 --output ./results/webcam_tiny.avi --model yolov4
```

### Changing The Tracking Classes
You can change which classes should tracked by modifying ``data/classes/tracking.names`` file. By default, it only tracks ``person`` and ``dog`` classes.

# Credits  

  * [hunglc007 Yolov4 TensorFlow Repo](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [nwojke DeepSort Repo](https://github.com/nwojke/deep_sort)
  * [TheAIGuy DeepSort Repo](https://github.com/theAIGuysCode/yolov4-deepsort)
