from pathlib import Path

from python_code.data_loaders.trajectory_loader.trajectory_csv_io import load_trajectory_csv
from python_code.data_loaders.trajectory_loader.trajectory_dataset import TrajectoryType

if __name__ == "__main__":
    recording_path = r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1"
    clip_name = "0m_37s-1m_37"
    left_eye_csv_path = r"clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    left_eye_csv_path = Path(recording_path) / 'clips'/clip_name/ left_eye_csv_path
    if not left_eye_csv_path.exists() and left_eye_csv_path.is_file():
        raise ValueError(f"CSV does not exist: {left_eye_csv_path}")

    eye_trajectories = load_trajectory_csv(filepath=left_eye_csv_path,
                                           trajectory_type=TrajectoryType.POSITION_2D,
                                           min_confidence=0.3)

    landmarks = {
        "p1": 0,
        "p2": 1,
        "p3": 2,
        "p4": 3,
        "p5": 4,
        "p6": 5,
        "p7": 6,
        "p8": 7,
        "tear_duct": 8,
        "outer_eye": 9,
    }

    connections = (
        #pupil outline
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        # additional connections
        (8, 9), # tear duct to outer eye
        (8, 0), # tear duct to p1
    )


