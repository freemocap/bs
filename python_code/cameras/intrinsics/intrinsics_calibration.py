from pathlib import Path
import cv2
import json
import numpy as np
from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker

np.set_printoptions(suppress=True)

# flag definitions -> https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
flags = (
    # cv2.CALIB_USE_INTRINSIC_GUESS +  #starting values for camera matrix are provided
    # cv2.CALIB_ZERO_TANGENT_DIST +  # p1 and p2 are set to zero and not changed
    cv2.CALIB_FIX_ASPECT_RATIO + # keeps fx/fy ratio constant
    cv2.CALIB_FIX_PRINCIPAL_POINT #+ # doesn't change center point
    # cv2.CALIB_FIX_FOCAL_LENGTH
    # cv2.CALIB_RATIONAL_MODEL  # solves for k4, k5, k6 (adds additional points to distortion coefficients)
    # cv2.CALIB_THIN_PRISM_MODEL
    # cv2.CALIB_TILTED_MODEL
)  # TODO: this doesn't need to be a global, just setting it here for now
# TODO: I've mostly narrowed down to flags we probably don't need. We may be able to skip this part.


def setup_7x5_tracker() -> CharucoTracker:
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    return CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )

def setup_5x3_tracker() -> CharucoTracker:
    charuco_squares_x_in = 5
    charuco_squares_y_in = 3
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    return CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )


def save_corrected_video(
    input_video_path: Path,
    output_video_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    print("Saving corrected video...")
    cap = cv2.VideoCapture(str(input_video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    include_all_pixels = 1 # 1: all pixels are retained with some extra black in margins, 0: only valid pixels are shown
    # we can do 1 here because we crop it down anyways

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), include_all_pixels, (width, height))
    x, y, w, h = roi

    out = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (w, h),
    )

    while True:
        ret, image = cap.read()

        if not ret:
            break

        corrected_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        cropped_corrected_image = corrected_image[y:y+h, x:x+w]

        cv2.imshow("Corrected Image", cropped_corrected_image)
        cv2.waitKey(1)
        out.write(cropped_corrected_image)

    cap.release()
    out.release()

    print(f"Corrected video saved to {output_video_path}")


def run_intrinsics(input_video_path: Path, output_video_path: Path, charuco_tracker: CharucoTracker = setup_7x5_tracker()):
    board = charuco_tracker.board
    detector = charuco_tracker.charuco_detector

    minimum_corners = 8 # documentation says >= 6, but sometimes throws an error with frames with fewer corners

    all_charuco_corners = []
    all_charuco_ids = []

    all_image_points = []
    all_object_points = []

    cap = cv2.VideoCapture(str(input_video_path))

    # opencv convention is (width, height)
    image_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    frame_number = -1
    while True:
        ret, image = cap.read()
        frame_number += 1

        if not ret:
            break

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_object_points = np.array([])
        current_image_points = np.array([])

        current_charuco_corners, current_charuco_ids, _, _ = detector.detectBoard(
            gray_image
        )

        if (
            current_charuco_corners is not None and len(current_charuco_corners) >= minimum_corners
        ):
            current_object_points, current_image_points = board.matchImagePoints(
                current_charuco_corners,
                current_charuco_ids,
                current_object_points,
                current_image_points,
            )

            if not current_image_points.any() or not current_object_points.any():
                print(f"Point matching failed for frame {frame_number}.")
            else:
                print(f"Frame captured: {frame_number}")

                all_charuco_corners.append(current_charuco_corners)
                all_charuco_ids.append(current_charuco_ids)
                all_image_points.append(current_image_points)
                all_object_points.append(current_object_points)

    print("Debugging Info")
    print(f"\tlength of all_charuco_corners: {len(all_charuco_corners)}")
    print(f"\tlength of all_charuco_ids: {len(all_charuco_ids)}")
    print(f"\tlength of all_image_points: {len(all_image_points)}")
    print(f"\tlength of all_object_points: {len(all_object_points)}")

    # TODO: filter for "better" views of the charuco somehow, and possibly filter for "most different" views to get variety

    if len(all_object_points) > 1000:
        sample = (len(all_object_points) // 1000) + 1
        print(f"Too many frames captured, sampling every {sample} frames")
        all_object_points = all_object_points[:1000]
        all_image_points = all_image_points[:1000]

    ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    distortion_coefficients = distortion_coefficients[0]

    print(f"Camera matrix: {camera_matrix}")
    print(f"Distortion coefficients: {distortion_coefficients}")

    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coefficients.tolist(),
    }
    json_output_path = output_video_path.with_name(f"{input_video_path.stem}_intrinsics.json")
    with open(str(json_output_path), "w") as f:
        json.dump(calibration_data, f, indent=4)

    cap.release()

if __name__ == "__main__":
    input_video_path = Path(
        "/home/scholl-lab/recordings/session_2025-08-06/calibration/raw_videos/24908832.mp4"
    )
    output_video_path = input_video_path.with_name(
        f"{input_video_path.stem}_corrected.mp4"
    )
    charuco_tracker = setup_5x3_tracker()  # Use the 5x3 charuco board
    # charuco_tracker = setup_7x5_tracker()  # Use the 7x5 charuco board

    run_intrinsics(
        input_video_path=input_video_path, output_video_path=output_video_path, charuco_tracker=charuco_tracker
    )
