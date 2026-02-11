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


def plot_horizontal_position_vs_azimuth_across_eyes(
    left_eye_kinematics: FerretEyeKinematics, 
    right_eye_kinematics: FerretEyeKinematics,
    left_gaze_df: pl.DataFrame, 
    right_gaze_df: pl.DataFrame,
    save_path):
    left_eye_horizontal_position = left_eye_kinematics.tracked_pupil.pupil_center_mm[:, 0]
    right_eye_horizontal_position = right_eye_kinematics.tracked_pupil.pupil_center_mm[:, 0]

    left_gaze_azimuth_degrees = left_gaze_df["Azimuth_degrees"].to_numpy()
    right_gaze_azimuth_degrees = right_gaze_df["Azimuth_degrees"].to_numpy()

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

    plot_horizontal_position_vs_azimuth_across_eyes(
        left_eye_kinematics=left_eye,
        right_eye_kinematics=right_eye,
        left_gaze_df=left_orientation_df,
        right_gaze_df=right_orientation_df,
        save_path=output_dir / "horizontal_position_vs_azimuth_across_eyes.png"
    )