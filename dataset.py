import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from data_calibration import calibrate_amplitude, dwn_noise, hampel

# -------------------------
# PATH CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114


# -------------------------
# DATA LOADING
# -------------------------
def read_csi_data_from_csv(path_to_csv, is_five_hhz=False, antenna_pairs=4):

    if not os.path.exists(path_to_csv):
        raise FileNotFoundError(f"Missing file: {path_to_csv}")

    data = pd.read_csv(path_to_csv, header=None).values

    subcarries_num = SUBCARRIES_NUM_FIVE_HHZ if is_five_hhz else SUBCARRIES_NUM_TWO_HHZ

    amplitudes = data[:, subcarries_num * 1:subcarries_num * (1 + antenna_pairs)]
    phases = data[:, subcarries_num * (1 + antenna_pairs):subcarries_num * (1 + 2 * antenna_pairs)]

    return amplitudes, phases


def read_labels_from_csv(path_to_csv):

    if not os.path.exists(path_to_csv):
        raise FileNotFoundError(f"Missing file: {path_to_csv}")

    data = pd.read_csv(path_to_csv, header=None).values
    return data[:, 1]


def discover_dataset_paths():
    """
    Automatically finds all dataset folders containing data.csv and label.csv
    """
    paths = []

    for root, dirs, files in os.walk(DATASET_FOLDER):
        if "data.csv" in files and "label.csv" in files:
            paths.append(root)

    if len(paths) == 0:
        raise RuntimeError("No dataset folders found. Expected structure: dataset/*/*/data.csv")

    return paths


def read_all_data_from_files(paths, is_five_hhz=True, antenna_pairs=4):

    final_amplitudes, final_phases, final_labels = [], [], []

    for path in paths:
        amplitudes, phases = read_csi_data_from_csv(os.path.join(path, "data.csv"), is_five_hhz, antenna_pairs)
        labels = read_labels_from_csv(os.path.join(path, "label.csv"))

        amplitudes, phases = amplitudes[:-1], phases[:-1]

        final_amplitudes.append(amplitudes)
        final_phases.append(phases)
        final_labels.append(labels)

    return (
        np.concatenate(final_amplitudes),
        np.concatenate(final_phases),
        np.concatenate(final_labels),
    )


# -------------------------
# DATASET CLASS
# -------------------------
class CSIDataset(Dataset):

    def __init__(self, window_size=32, step=1):

        paths = discover_dataset_paths()

        self.amplitudes, self.phases, self.labels = read_all_data_from_files(paths)

        # Normalize amplitude
        self.amplitudes = calibrate_amplitude(self.amplitudes)

        # Noise filtering
        data_len = self.amplitudes.shape[0]
        for i in range(self.amplitudes.shape[1]):
            self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[:data_len]

        # Label mapping
        self.class_to_idx = {
            "standing": 0,
            "walking": 1,
            "get_down": 2,
            "sitting": 3,
            "get_up": 4,
            "lying": 5,
            "no_person": 6
        }

        self.window = window_size
        self.step = step

    def __getitem__(self, idx):

        idx = idx * self.step
        xs = []

        for i in range(idx, idx + self.window):
            xs.append(self.amplitudes[i])

        return np.array(xs), self.class_to_idx[self.labels[idx + self.window - 1]]

    def __len__(self):
        return int((self.labels.shape[0] - self.window) // self.step) + 1


# -------------------------
# TEST RUN
# -------------------------
if __name__ == '__main__':
    dataset = CSIDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    for x, y in loader:
        print("Batch shape:", x.shape)
        print("Labels:", y.shape)
        break
