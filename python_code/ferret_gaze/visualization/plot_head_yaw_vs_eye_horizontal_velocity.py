import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeballKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics


def plot_head_yaw_vs_eye_horizontal_velocity(skull: SkullKinematics,
                                             left_eye: EyeballKinematics,
                                             right_eye: EyeballKinematics) -> None:

    head_yaw_velocity = skull.head_yaw_velocity_deg_s
    left_eye_horizontal_velocity_deg_s = np.rad2deg(left_eye.eye_horizontal_velocity_rad_s)
    right_eye_horizontal_velocity_deg_s = np.rad2deg(right_eye.eye_horizontal_velocity_rad_s)
    # Create a scatter plot with a regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=head_yaw_velocity, y=left_eye_horizontal_velocity_deg_s, label='Left Eye', scatter_kws={'alpha': 0.5})
    sns.regplot(x=head_yaw_velocity, y=right_eye_horizontal_velocity_deg_s, label='Right Eye', scatter_kws={'alpha': 0.5}, color='orange')

    # Set labels and title
    plt.xlabel('Head Yaw (degrees)')
    plt.ylabel('Eye Horizontal Velocity (degrees/s)')
    plt.title('Head Yaw vs Eye Horizontal Velocity')
    plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()
