"""Test serialization for kinematics data."""
import sys
import tempfile
from pathlib import Path

import numpy as np

from python_code.ferret_gaze.kinematics_core.quaternion_model import Quaternion

# Add uploads to path for imports
sys.path.insert(0, "/mnt/user-data/uploads")

from reference_geometry_model import ReferenceGeometry


def create_test_geometry() -> ReferenceGeometry:
    """Create a simple test reference geometry."""
    return ReferenceGeometry(
        **{
            "units": "mm",
            "coordinate_frame": {
                "origin_markers": ["left_eye", "right_eye"],
                "x_axis": {"markers": ["nose"], "type": "exact"},
                "y_axis": {"markers": ["left_eye"], "type": "approximate"},
            },
            "markers": {
                "nose": {"x": 18.0, "y": 0.0, "z": 0.0},
                "left_eye": {"x": 0.0, "y": 12.0, "z": 0.0},
                "right_eye": {"x": 0.0, "y": -12.0, "z": 0.0},
                "back": {"x": -20.0, "y": 0.0, "z": 5.0},
            },
        }
    )


def test_serialization() -> None:
    """Test that kinematics can be serialized correctly."""
    print("\n=== Testing Serialization ===")

    # Import serialization functions
    sys.path.insert(0, "/home/claude")
    from kinematics_serialization import (
        kinematics_to_tidy_dataframe,
        save_kinematics,
    )

    # Create test data manually (mimicking RigidBodyKinematics structure)
    class MockKinematics:
        def __init__(self) -> None:
            self.name = "test_body"
            self.reference_geometry = create_test_geometry()
            number_of_frames = 10
            self.n_frames = number_of_frames
            self.timestamps = np.linspace(0.0, 1.0, number_of_frames)
            angles = np.linspace(0, np.pi / 2, number_of_frames)
            self.position_xyz = np.column_stack(
                [100 * np.cos(angles), 100 * np.sin(angles), np.zeros(number_of_frames)]
            )
            self.velocity_xyz = np.gradient(self.position_xyz, self.timestamps, axis=0)
            self.orientations = [
                Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2)) for angle in angles
            ]
            self.angular_velocity_global = np.zeros((number_of_frames, 3))
            self.angular_velocity_global[:, 2] = np.gradient(angles, self.timestamps)
            self.angular_velocity_local = self.angular_velocity_global.copy()

        @property
        def duration(self) -> float:
            return float(self.timestamps[-1] - self.timestamps[0])

        @property
        def keypoint_names(self) -> list[str]:
            return list(self.reference_geometry.markers.keys())

        def get_keypoint_trajectory(self, name: str) -> "MockTrajectory":
            local_position = self.reference_geometry.markers[name].to_array()
            world_positions = np.zeros((self.n_frames, 3), dtype=np.float64)
            for index, quaternion in enumerate(self.orientations):
                world_positions[index] = self.position_xyz[index] + quaternion.rotate_vector(local_position)
            return MockTrajectory(values=world_positions)

    class MockTrajectory:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

    kinematics = MockKinematics()

    # Test 1: DataFrame export
    print("  Testing kinematics_to_tidy_dataframe...")
    dataframe = kinematics_to_tidy_dataframe(kinematics=kinematics)
    assert len(dataframe) > 10  # Tidy format has many rows
    assert "trajectory" in dataframe.columns
    assert "component" in dataframe.columns
    assert "reference_frame" in dataframe.columns
    assert "value" in dataframe.columns
    print(f"    ✓ Tidy DataFrame has {len(dataframe)} rows, {len(dataframe.columns)} columns")

    # Test 2: Full save cycle
    print("  Testing save_kinematics...")
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory = Path(temporary_directory)
        reference_geometry_path, kinematics_csv_path = save_kinematics(
            kinematics=kinematics, output_directory=output_directory, include_keypoints=True
        )

        assert reference_geometry_path.exists(), f"Geometry file not created: {reference_geometry_path}"
        assert kinematics_csv_path.exists(), f"CSV file not created: {kinematics_csv_path}"

        # Verify CSV has tidy format with keypoint rows
        import pandas as pd

        loaded_dataframe = pd.read_csv(kinematics_csv_path)
        assert "trajectory" in loaded_dataframe.columns
        assert "component" in loaded_dataframe.columns
        
        # Check keypoints are present
        keypoint_rows = loaded_dataframe[loaded_dataframe["trajectory"].str.startswith("keypoint__")]
        assert len(keypoint_rows) > 0, "No keypoint rows found"

        # Verify reference geometry JSON
        loaded_geometry = ReferenceGeometry.from_json_file(path=reference_geometry_path)
        assert loaded_geometry.units == "mm"
        assert "nose" in loaded_geometry.markers

        print(f"    ✓ Files saved correctly")
        print(f"      - {reference_geometry_path.name}")
        print(f"      - {kinematics_csv_path.name} ({len(loaded_dataframe)} rows)")

    print("\n  ✓ All serialization tests passed!")


if __name__ == "__main__":
    test_serialization()
