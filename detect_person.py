import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

--------------------------------------------------
Load YOLO model once (not every image)
--------------------------------------------------

YOLO_CFG = "./yolo_data/yolov3.cfg"
YOLO_WEIGHTS = "./yolo_data/yolov3.weights"
YOLO_NAMES = "./yolo_data/coco.names"

CONFIDENCE = 0.5

with open(YOLO_NAMES) as f:
LABELS = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

--------------------------------------------------
Detect person in one image
--------------------------------------------------

def detect_person_box(image_path):
image = cv2.imread(image_path)
if image is None:
return [0, 0, 0, 0]

h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(layer_names)

best_box = [0, 0, 0, 0]
best_conf = 0

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if LABELS[class_id] == "person" and confidence > best_conf:
            box = detection[:4] * np.array([w, h, w, h])
            centerX, centerY, width, height = box.astype("int")
            x = int(centerX - width / 2)
            y = int(centerY - height / 2)

            best_box = [x, y, width, height]
            best_conf = confidence

return best_box
--------------------------------------------------
Process full folder
--------------------------------------------------

def process_dataset(dataset_path):
image_folder = os.path.join(dataset_path, "images")
output_csv = os.path.join(dataset_path, "label_boxes.csv")

images = sorted(os.listdir(image_folder))
images = [os.path.join(image_folder, img) for img in images]

all_boxes = []

for img in tqdm(images, desc="Detecting humans"):
    box = detect_person_box(img)
    all_boxes.append(box)

df = pd.DataFrame(all_boxes, columns=["x", "y", "width", "height"])
df.to_csv(output_csv, index=False)

print("Saved:", output_csv)
--------------------------------------------------
Run
--------------------------------------------------

if name == "main":
DATASET_PATH = "./dataset/session1" # change per recording
process_dataset(DATASET_PATH)