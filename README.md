# Demosaicing a Time-Varying Exposures Array for Snapshot HDR Imaging

<img src="static/teaser.PNG" width="40%" height="50%" title="teaser"></img>

### Challenges

The spatially-varying exposure input has low spatial resolution and different motion blur for each exposure level (b).

Our network restores HDR images (c) from [quad Bayer patterned sensor](https://semiconductor.samsung.com/image-sensor/mobile-image-sensor/isocell-gn1/) images (a) in an end-to-end manner.
We jointly solve demosaicing and deblurring problems to achieve a high-quality snapshot HDR image from the quad-Bayer pattern.

Also, we create a dataset of quad Bayer sensor input with varying exposures and colors using the existing HDR video dataset.

[Presentation file](https://drive.google.com/file/d/1sJswcXjly7GI8QgXUr1gs9F5Adcq33TW/view?usp=share_link) (Google Drive)

## Results

### Quantitative Results For Synthetic Dataset

<img src="static/quantitative_syn.PNG" width="60%" height="50%" title="quantitative_syn"></img>

### Qualitative Results For Synthetic Dataset

<img src="static/qualitative_syn.PNG" width="90%" height="50%" title="qualitative_syn"></img>

### Qualitative Results For Real-world Dataset

<img src="static/qualitative_real.PNG" width="90%" height="50%" title="qualitative_real"></img>

Quantitative and qualitative results of our model. Our model outperforms baseline methods in terms of color reconstruction and denoising, particularly in the area with strong motion blur.

## Methods

<img src="static/network.PNG" width="100%" height="50%" title="network"></img>

Our method consists of two main modules: 

(a) HDR feature extraction module: demosaicing and upscaling

(b) HDR feature registration module: motion blur registration

## Installation

This repository is built in Pytorch 1.9.0 and tested on Ubuntu 16.04 enviornment (Python3.7, CUDA10.2, cuDNN7.6).

Or you can just pull official docker image from Dockerhub and istall dependencies:
```
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
git clone https://github.com/kimjw0623/HDR_Imaging.git
pip install -r "requirements.txt"
```

Follow these instructions:
1. Clone our repository
```
git clone https://github.com/kimjw0623/HDR_Imaging.git
cd HDR_Imaging
```

2. Make conda enviornment
```
conda create -n pytorch190 python=3.7
conda activate pytorch190
```

3. Install dependencies
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r "requirements.txt"
```

## Demo
To test pre-trained models, run below code:
```
python test.py
```
