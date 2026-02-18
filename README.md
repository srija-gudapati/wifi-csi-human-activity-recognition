**WiFi CSI Human Activity Recognition**
A deep learning based system that recognizes human activities using WiFi Channel State Information (CSI) signals instead of cameras or wearable sensors.
The model learns motion patterns from wireless signal distortions caused by human movement.

**Activities Detected**

• Standing

• Walking

• Sitting

• Lying

• Get Up

• Get Down

• No Person

**Pipeline**

• Collect raw CSI signal data

• Calibrate amplitude & phase

• Noise filtering (Hampel + Wavelet)

• Feature extraction using PCA

• Train deep learning model (LSTM / InceptionTime)

• Predict activity

**Model Architecture**

LSTM based sequence classifier

InceptionTime temporal CNN (optional)

Sliding window time-series classification


**Installation**

pip install -r requirements.txt

**Run**

python train.py

python evaluate.py

**Applications**

• Elderly fall detection

• Smart homes

• Contactless monitoring

• Healthcare environments
