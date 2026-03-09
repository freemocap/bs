import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path
from scipy.spatial.transform import Rotation as R

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory
from python_code.kinematics_core.kinematics_serialization import load_kinematics
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics


def plot_horizontal_velocity_vs_local_head_yaw_velocity(
    left_eye_kinematics: FerretEyeKinematics, 
    right_eye_kinematics: FerretEyeKinematics,
    skull_kinematics: RigidBodyKinematics,
    save_path):
    frame_start = 0
    frame_end = 1200

    left_eye_horizontal_position = left_eye_kinematics.tracked_pupil.pupil_center_mm[frame_start:frame_end, 0]
    right_eye_horizontal_position = right_eye_kinematics.tracked_pupil.pupil_center_mm[frame_start:frame_end, 0]

    angular_velocity_local = skull_kinematics.angular_velocity_local[frame_start:frame_end, :]

    left_timestamps = left_eye_kinematics.eyeball.timestamps
    right_timestamps = right_eye_kinematics.eyeball.timestamps

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Left and Right Horizontal Eye Position vs Gaze Azimuth")
    ax1.scatter(left_eye_horizontal_position, left_gaze_azimuth_degrees, alpha=0.3)
    ax1.set_title("Left Eye Horizontal Position vs Azimuth")
    ax1.set_xlabel("Horizontal Position (mm)")
    ax1.set_ylabel("Azimuth (degrees)")

    ax2.scatter(right_eye_horizontal_position, right_gaze_azimuth_degrees, alpha=0.3)
    ax2.set_title("Right Eye Horizontal Position vs Azimuth")
    ax2.set_xlabel("Horizontal Position (mm)")
    ax2.set_ylabel("Azimuth (degrees)")

    # plot eye Azimuth over time
    ln3 = ax3.plot(left_timestamps, left_gaze_azimuth_degrees, color="blue", alpha=0.7, label="Azimuth", linewidth=2)
    twin_ax3 = ax3.twinx()
    ln3_twin = twin_ax3.plot(left_timestamps, left_eye_horizontal_position, color="orange", alpha=1, label="Horizontal Position", linewidth=0.5)
    ax3.set_title("Azimuth and Horizontal Position")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Azimuth (degrees)", color="blue")
    twin_ax3.set_ylabel("Horizontal Position (mm)", color="orange")

    ln4 = ax4.plot(right_timestamps, right_gaze_azimuth_degrees, color="blue", alpha=0.7, label="Azimuth", linewidth=2)
    twin_ax4 = ax4.twinx()
    ln4_twin = twin_ax4.plot(right_timestamps, right_eye_horizontal_position, color="orange", alpha=1, label="Horizontal Position", linewidth=0.5)
    ax4.set_title("Azimuth and Horizontal Position")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Azimuth (degrees)", color="blue")
    twin_ax4.set_ylabel("Horizontal Position (mm)", color="orange")

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

    skull = load_kinematics(
        reference_geometry_path=analyzable_output_dir / "skull_kinematics" / "skull_reference_geometry.json",
        kinematics_csv_path=analyzable_output_dir / "skull_kinematics" / "skull_kinematics.csv",
    )

    output_dir = analyzable_output_dir / "test_plots"
    output_dir.mkdir(exist_ok=True)

    plot_horizontal_velocity_vs_local_head_yaw_velocity(
        left_eye_kinematics=left_eye,
        right_eye_kinematics=right_eye,
        skull_kinematics=skull,
        save_path=output_dir / "horizontal_velocity_vs_local_head_yaw_velocity.png"
    )