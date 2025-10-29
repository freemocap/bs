from datetime import datetime
from pathlib import Path
import numpy as np
import toml
import rerun as rr

def log_camera(camera_name: str, position: np.ndarray, orientation: np.ndarray, intrinsics: np.ndarray):
    print(f"Camera: {camera_name}")
    print(f"position: {position}")
    print(f"orientation: {orientation}")

    rr.log(
        f"/cameras/{camera_name}",
        rr.Transform3D(translation=position, mat3x3=orientation),
        static=True
    )
    rr.log(
        f"/cameras/{camera_name}",
        rr.Pinhole(
            resolution=(1024, 1024),
            image_from_camera=intrinsics,
            camera_xyz=rr.ViewCoordinates.RDF
        ), 
        static=True
    )
    # TODO: add images like rr.log(f"/cameras/{camera_name}", rr.EncodedImage(path=....))

def log_cameras(calibration: dict):
    for key, value in calibration.items():
        if key.startswith("cam_"):
            log_camera(camera_name=value["name"], position=value["world_position"], orientation=value["world_orientation"], intrinsics=value["matrix"])

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.groundplane_and_origin import log_groundplane_and_origin
    calibration_toml = Path("/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/calibration/session_2025_07_11_calibration_camera_calibration.toml")
    calibration = toml.load(calibration_toml)
    recording_string = (
        f"camera_test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)
    log_cameras(calibration)
    log_groundplane_and_origin()