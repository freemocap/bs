import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path
from scipy.spatial.transform import Rotation as R

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory

def get_rotations_from_gaze_df(gaze_df: pl.DataFrame) -> pl.DataFrame:
    orientation_df = gaze_df.filter(pl.col("trajectory") == "orientation")
    wide_orientation_df = orientation_df.pivot("component", index="frame", values="value")
    print(wide_orientation_df.head(5))
    quat_np = wide_orientation_df.select(["x", "y", "z", "w"]).to_numpy()   # shape (n_frames, 4)
    print(quat_np.shape)
    print(quat_np[0])
    rotations = R.from_quat(quat_np)
    euler_angles = rotations.as_euler("xyz", degrees=True)
    wide_orientation_df = wide_orientation_df.with_columns(
        pl.Series("roll_degrees", euler_angles[:, 0]),
        pl.Series("Elevation_degrees", euler_angles[:, 1]),
        pl.Series("Azimuth_degrees", euler_angles[:, 2]),
    )

    return wide_orientation_df


def plot_eye_position_vs_gaze_angle(eye_kinematics: FerretEyeKinematics, gaze_df: pl.DataFrame, save_path):
    eye_horizontal_position = eye_kinematics.tracked_pupil.pupil_center_mm[:, 0]
    eye_vertical_position = eye_kinematics.tracked_pupil.pupil_center_mm[:, 1]

    gaze_Azimuth_degrees = gaze_df["Azimuth_degrees"].to_numpy()
    gaze_Elevation_degrees = gaze_df["Elevation_degrees"].to_numpy()

    timestamps = eye_kinematics.eyeball.timestamps

    eye_name = save_path.name.split("_")[0].capitalize()

    # make fig with 6 subplots in a 3 rows by 2 columns grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{eye_name} Eye Position vs Gaze Angle")
    ax1.scatter(eye_horizontal_position, gaze_Azimuth_degrees, alpha=0.3)
    ax1.set_title("Horizontal Position vs Azimuth")
    ax1.set_xlabel("Horizontal Position (mm)")
    ax1.set_ylabel("Azimuth (degrees)")

    ax2.scatter(eye_vertical_position, gaze_Elevation_degrees, alpha=0.3)
    ax2.set_title("Vertical Position vs Elevation")
    ax2.set_xlabel("Vertical Position (mm)")
    ax2.set_ylabel("Elevation (degrees)")

    # plot eye Azimuth over time
    ln3 = ax3.plot(timestamps, gaze_Azimuth_degrees, color="blue", alpha=0.7, label="Azimuth", linewidth=2)
    twin_ax3 = ax3.twinx()
    ln3_twin = twin_ax3.plot(timestamps, eye_horizontal_position, color="orange", alpha=1, label="Horizontal Position", linewidth=0.5)
    ax3.set_title("Azimuth and Horizontal Position")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Azimuth (degrees)", color="blue")
    twin_ax3.set_ylabel("Horizontal Position (mm)", color="orange")

    # plot eye Elevation over time
    ln4 = ax4.plot(timestamps, gaze_Elevation_degrees, color="blue", alpha=0.7, label="Elevation", linewidth=2)
    twin_ax4 = ax4.twinx()
    ln4_twin = twin_ax4.plot(timestamps, eye_vertical_position, color="orange", alpha=1, label="Vertical Position", linewidth=0.5)
    ax4.set_title("Elevation and Vertical Position")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Elevation (degrees)", color="blue")
    twin_ax4.set_ylabel("Vertical Position (mm)", color="orange")

    plt.tight_layout()
    plt.show()
    # plt.savefig(save_path)


if __name__ == "__main__":
    analyzable_output_dir = Path("/Users/philipqueen/analyzable_output")
    left_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "left_eye_kinematics",
        eye_name="left_eye",
    )

    left_eye_gaze_df = pl.read_csv(analyzable_output_dir / "gaze_kinematics" / "left_gaze_kinematics.csv")
    left_orientation_df = get_rotations_from_gaze_df(left_eye_gaze_df)

    right_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "right_eye_kinematics",
        eye_name="right_eye",
    )

    right_eye_gaze_df = pl.read_csv(analyzable_output_dir / "gaze_kinematics" / "right_gaze_kinematics.csv")
    right_orientation_df = get_rotations_from_gaze_df(right_eye_gaze_df)

    output_dir = analyzable_output_dir / "test_plots"
    output_dir.mkdir(exist_ok=True)

    plot_eye_position_vs_gaze_angle(left_eye, left_orientation_df, output_dir / "left_eye_position_vs_gaze_angle.png")
    plot_eye_position_vs_gaze_angle(right_eye, right_orientation_df, output_dir / "right_eye_position_vs_gaze_angle.png")