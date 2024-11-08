# Commands

1. Setup
```bash
conda create -n mplug_owl python=3.10
conda activate mplug_owl
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

2. Inference
Change video_path in test.py to a small mp4 file (can use one from MSR VTT)