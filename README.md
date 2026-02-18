WiFi CSI Human Activity Recognition

This project predicts human activities using WiFi signals instead of cameras or wearable sensors. The idea is that when a person moves, sits, walks, or lies down, they slightly disturb wireless signals in the room. Those disturbances are captured as CSI (Channel State Information) data and used as input to deep learning models to recognize the activity.

The system works in three main stages: preprocessing the raw CSI signal, extracting useful features, and training neural networks to classify activities. The raw amplitude and phase signals are noisy, so filtering, calibration, and PCA are applied before feeding them into the model. The final features are sequences of signal patterns rather than images or sensor readings.

Multiple deep learning architectures were implemented including LSTM and InceptionTime based models. The LSTM captures temporal patterns in human movement while the Inception model learns local signal variations across subcarriers. The model predicts activities such as standing, walking, sitting, lying, getting up, getting down, and no person present.

The dataset is organized room-wise and processed into sliding windows so that each prediction depends on a short time history of signal changes instead of a single frame. This makes the predictions more stable and realistic for real-world usage.

To improve reliability, YOLO-based human detection was also used to generate bounding box labels from recorded images. This allows filtering of signal samples when no person is actually present, preventing false learning.

This project demonstrates how wireless signals alone can be used for activity recognition, enabling privacy-preserving monitoring systems for smart homes, elderly care, and human-computer interaction.

How to Run

Install dependencies

pip install -r requirements.txt


Train model

python train.py


Evaluate model

python evaluate.py

Tech Stack

Python, PyTorch, NumPy, Pandas, OpenCV, PCA, LSTM, InceptionTime, YOLOv3
