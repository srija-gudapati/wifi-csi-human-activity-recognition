from enum import Enum
import os
import numpy as np
import pandas as pd

# -------------------------
# ACTIVITY DEFINITIONS
# -------------------------
class Activity(Enum):
    NO_PERSON = "no_person"
    STANDING = "standing"
    WALKING = "walking"
    SITTING = "sitting"
    LYING = "lying"
    GET_UP = "get_up"
    GET_DOWN = "get_down"


# -------------------------
# LABEL FUNCTION
# -------------------------
def apply_activity(labels, start, end, activity: Activity):
    labels[start:end] = activity.value


# -------------------------
# MAIN LABELING TOOL
# -------------------------
def label_recording(folder_path, segments):
    """
    segments = list of tuples:
    [
        (start_index, end_index, Activity.STANDING),
        (end_index, end_index, Activity.WALKING)
    ]
    """

    data_path = os.path.join(folder_path, "data.csv")
    label_path = os.path.join(folder_path, "label.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data.csv found in {folder_path}")

    data = pd.read_csv(data_path, header=None)
    labels = np.empty(data.shape[0], dtype=object)

    # default
    labels[:] = Activity.NO_PERSON.value

    for start, end, activity in segments:
        apply_activity(labels, start, end, activity)

    pd.DataFrame(labels).to_csv(label_path, index=False, header=False)
    print(f"Saved labels -> {label_path}")


# -------------------------
# EXAMPLE USAGE
# -------------------------
if __name__ == "__main__":

    example_segments = [
        (0, 200, Activity.STANDING),
        (200, 400, Activity.WALKING),
        (400, 600, Activity.SITTING),
        (600, 800, Activity.LYING),
    ]

    label_recording("dataset/session1", example_segments)
