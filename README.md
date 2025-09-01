# Sport CV Pipeline — NVIDIA RTX 5070 Ti Showcase

A modular computer-vision stack for real‑time sports analytics, engineered with NVIDIA to demonstrate the raw throughput of the RTX 5070 Ti. The system blends YOLOv9c detection, ByteTrack association, Kalman‑filter ball estimation, and a suite of domain‑specific analytics (speed & distance, ball possession, team assignment) under a TensorRT‑accelerated inference core.

This pipeline was built for an NVIDIA showcase on the RTX 5070 Ti. With TensorRT‑optimized YOLOv9c, the system achieves <10 ms per-frame latency, enabling real-time analytics on consumer GPUs.
---

## Features

- **High‑precision detection & tracking**  
  YOLOv9c for players/officials/ball, fused with supervision’s ByteTrack for ID‑stable multi-object tracking and a custom Kalman module that interpolates missing ball detections.

- **Physics‑aware analytics**  
  Homography‑based view transform, per‑player speed & distance estimation, automatic team assignment, and per‑frame ball possession metrics.

- **Realtime HUD & overlays**  
  Toggleable visualization of halos, camera-motion vectors, speed/distance readouts, and ball-possession indicators, rendered at 30 FPS on 1080p footage.

- **Training + Deployment**  
  End‑to‑end workflow: fine‑tune YOLOv9c, export to TensorRT, and benchmark latencies in the same repository.

---

## TensorRT on RTX 5070 Ti

![Latency Plot](docs/latency_5070ti.png)

---


---

## Setup

```bash
git clone https://github.com/yourname/sport-cv-pipeline.git
cd sport-cv-pipeline
pip install -r requirements.txt

TensorRT 10.12 GA is required for deployment:
set PATH=C:\Projects\TensorRT-10.12\TensorRT-10.12.0.36\lib;%PATH%
set PYTHONPATH=C:\Projects\TensorRT-10.12\TensorRT-10.12.0.36\python

Usage:
python -m train.train_yolo --cfg configs/train.yaml
python -m train.export_trt --weights runs/train/best.pt --batch 4 --fp16
python src/yolo_benchmark.py --video data/test_clip.mp4 --weights models/128060ep.pt --batch 4



