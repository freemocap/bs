import numpy as np
import pandas as pd
from pathlib import Path

landmarks = {
    "nose": 0,
    "left_cam_tip": 1,
    "right_cam_tip": 2,
    "base": 3,
    "left_eye": 4,
    "right_eye": 5,
    "left_ear": 6,
    "right_ear": 7,
    "spine_t1": 8,
    "sacrum": 9,
    "tail_tip": 10,
    "center": 11,
}

def load_tidy_dataset(csv_path: Path, landmarks: dict[str, int], data_type: str = "optimized") -> np.ndarray:
    df = pd.read_csv(csv_path)

    num_frames = df["frame"].nunique()
    print(f"Loaded {num_frames} frames")
    num_markers = df["marker"].nunique()

    if num_markers != len(landmarks):
        raise ValueError(f"Expected {len(landmarks)} markers, but found {num_markers} in {csv_path}")

    data = np.zeros((num_frames, num_markers, 3))
    for i, marker in enumerate(landmarks.keys()):
        masked_df = df.query(f'marker == "{marker}" and data_type == "{data_type}"')

        if len(masked_df) != num_frames:
            raise ValueError(f"Expected {num_frames} frames for marker {marker}, but found {len(masked_df)}")

        data[:, i, 0] = masked_df["x"].values
        data[:, i, 1] = masked_df["y"].values
        data[:, i, 2] = masked_df["z"].values

    return data

if __name__ == "__main__":
    csv_path = Path("/Users/philipqueen/Documents/GitHub/bs/output/2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m_37s-1m_37s/tidy_trajectory_data.csv")
    data = load_tidy_dataset(csv_path, landmarks)
    print(data.shape)