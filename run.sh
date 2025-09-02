#!/bin/bash
source env/bin/activate
cd app
python main.py --video ../data/video.mov --rois ../data/rois.yaml --conf 0.55 --save_video ../data/out.mp4