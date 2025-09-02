#!/bin/bash
source env/bin/activate
python app/main.py --video data/video.mov --rois data/rois.yaml --conf 0.55 --save_video data/out.mp4