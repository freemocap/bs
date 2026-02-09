import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory

def plot_eye_position_vs_gaze_angle(eye: FerretEyeKinematics, save_path):
    eye_azimuth_degrees = eye.azimuth_degrees
    eye_elevation_degrees = eye.elevation_degrees
    eye_horizontal_position = eye.tracked_pupil.pupil_center_mm[:, 0]
    eye_vertical_position = eye.tracked_pupil.pupil_center_mm[:, 1]

    timestamps = eye.eyeball.timestamps

    eye_name = save_path.name.split("_")[0].capitalize()

    # make fig with 6 subplots in a 3 rows by 2 columns grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{eye_name} Eye Position vs Gaze Angle")
    ax1.scatter(eye_horizontal_position, eye_azimuth_degrees, alpha=0.3)
    ax1.set_title("Horizontal Position vs Azimuth")
    ax1.set_xlabel("Horizontal Position (mm)")
    ax1.set_ylabel("Azimuth (degrees)")

    ax2.scatter(eye_vertical_position, eye_elevation_degrees, alpha=0.3)
    ax2.set_title("Vertical Position vs Elevation")
    ax2.set_xlabel("Vertical Position (mm)")
    ax2.set_ylabel("Elevation (degrees)")

    # plot eye azimuth over time
    ln3 = ax3.plot(timestamps, eye_azimuth_degrees, color="blue", alpha=0.7, label="Azimuth", linewidth=2)
    twin_ax3 = ax3.twinx()
    ln3_twin = twin_ax3.plot(timestamps, eye_horizontal_position, color="orange", alpha=1, label="Horizontal Position", linewidth=0.5)
    ax3.set_title("Azimuth and Horizontal Position")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Azimuth (degrees)", color="blue")
    twin_ax3.set_ylabel("Horizontal Position (mm)", color="orange")

    # plot eye elevation over time
    ln4 = ax4.plot(timestamps, eye_elevation_degrees, color="blue", alpha=0.7, label="Elevation", linewidth=2)
    twin_ax4 = ax4.twinx()
    ln4_twin = twin_ax4.plot(timestamps, eye_vertical_position, color="orange", alpha=1, label="Vertical Position", linewidth=0.5)
    ax4.set_title("Elevation and Vertical Position")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Elevation (degrees)", color="blue")
    twin_ax4.set_ylabel("Vertical Position (mm)", color="orange")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":
    analyzable_output_dir = Path("/Users/philipqueen/analyzable_output")
    left_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "left_eye_kinematics",
        eye_name="left_eye",
    )

    right_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "right_eye_kinematics",
        eye_name="right_eye",
    )
    output_dir = analyzable_output_dir / "test_plots"
    output_dir.mkdir(exist_ok=True)

    plot_eye_position_vs_gaze_angle(left_eye, output_dir / "left_eye_position_vs_gaze_angle.png")
    plot_eye_position_vs_gaze_angle(left_eye, output_dir / "right_eye_position_vs_gaze_angle.png")