import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory

def plot_gaze_angle_between_sessions(eye_one: FerretEyeKinematics, eye_two: FerretEyeKinematics, save_path):
    eye_one_azimuth_degrees = eye_one.azimuth_degrees[1000:]
    eye_one_elevation_degrees = eye_one.elevation_degrees[1000:]
    eye_one_horizontal_position = eye_one.tracked_pupil.pupil_center_mm[1000:, 0]
    eye_one_vertical_position = eye_one.tracked_pupil.pupil_center_mm[1000:, 1]

    eye_one_timestamps = eye_one.eyeball.timestamps[1000:]

    eye_two_azimuth_degrees = eye_two.azimuth_degrees[1000:]
    eye_two_elevation_degrees = eye_two.elevation_degrees[1000:]
    eye_two_horizontal_position = eye_two.tracked_pupil.pupil_center_mm[1000:, 0]
    eye_two_vertical_position = eye_two.tracked_pupil.pupil_center_mm[1000:, 1]

    eye_two_timestamps = eye_two.eyeball.timestamps[1000:]

    eye_name = save_path.name.split("_")[0].capitalize()

    # make fig with 6 subplots in a 3 rows by 2 columns grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f"Gaze Angle Across Sessions")

    ax1.plot(eye_one_timestamps, eye_one_azimuth_degrees, color="blue", alpha=0.7, label="Azimuth", linewidth=2)
    ax1.plot(eye_one_timestamps, eye_one_elevation_degrees, color="orange", alpha=0.7, label="Elevation", linewidth=2)
    ax1.set_title("Ferret 1 Azimuth and Elevation")

    ax2.plot(eye_one_timestamps, eye_one_elevation_degrees, color="blue", alpha=0.7, label="Elevation", linewidth=2)
    ax2.plot(eye_two_timestamps, eye_two_elevation_degrees, color="orange", alpha=0.7, label="Elevation", linewidth=2)
    ax2.set_title("Ferret 2 Elevation")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Elevation (degrees)")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":
    first_eye_analyzable_output_dir = Path("/Users/philipqueen/analyzable_output")
    first_left_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=first_eye_analyzable_output_dir / "left_eye_kinematics",
        eye_name="left_eye",
    )

    first_right_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=first_eye_analyzable_output_dir / "right_eye_kinematics",
        eye_name="right_eye",
    )
    second_eye_analyzable_output_dir = Path("/Users/philipqueen/analyzable_output")
    second_left_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=second_eye_analyzable_output_dir / "left_eye_kinematics",
        eye_name="left_eye",
    )

    second_right_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=second_eye_analyzable_output_dir / "right_eye_kinematics",
        eye_name="right_eye",
    )
    output_dir = second_eye_analyzable_output_dir / "test_plots"
    output_dir.mkdir(exist_ok=True)

    plot_gaze_angle_between_sessions(eye_one=first_left_eye, eye_two=second_left_eye, save_path=output_dir / "left_gaze_angle_comparison.png")
    plot_gaze_angle_between_sessions(eye_one=first_right_eye, eye_two=second_right_eye, save_path=output_dir / "right_gaze_angle_comparison.png")