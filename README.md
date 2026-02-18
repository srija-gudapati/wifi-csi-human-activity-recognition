WiFi CSI Human Activity Recognition

A deep learning based system that recognizes human activities using WiFi Channel State Information (CSI) signals instead of cameras or wearable sensors.

The model learns motion patterns from wireless signal distortions caused by human movement.

Activities Detected

Standing

Walking

Sitting

Lying

Get Up

Get Down

No Person

Pipeline

Collect raw CSI signal data

Calibrate amplitude & phase

Noise filtering (Hampel + Wavelet)

Feature extraction using PCA

Train deep learning model (LSTM / InceptionTime)

Predict activity

Model Architecture

LSTM based sequence classifier

InceptionTime temporal CNN (optional)

Sliding window time-series classification

Project Structure
preprocess.py        → signal calibration & filtering
dataset.py           → dataloader & windowing
train.py             → model training
evaluate.py          → testing
models/              → deep learning architectures
experiments/         → CSI visualizations
yolo_weights/        → person detection support

Installation
pip install -r requirements.txt

Train
python train.py

Evaluate
python evaluate.py

Key Idea

Human movement slightly disturbs WiFi signals.
By learning these distortions over time, the model can recognize activities without cameras — enabling privacy-preserving indoor sensing.

Applications

Elderly fall detection

Smart homes

Contactless monitoring

Healthcare environments
