"""Main entry point for eye tracking optimization."""

import numpy as np
from pathlib import Path

from eye_data_loader import EyeTrackingData
from python_code.pyceres_solvers.eye_tracking.camera_model import PUPIL_LABS_CAMERA
from python_code.pyceres_solvers.eye_tracking.eye_model_io import save_full_results
from python_code.pyceres_solvers.eye_tracking.eye_pyceres_bundle_adjustment import OptimizationResult, OptimizationConfig, \
    optimize_eye_tracking_data, EyeModel
from python_code.pyceres_solvers.eye_tracking.eye_savers import EyeTrackingResults


def run_eye_tracking(*, csv_path: Path, output_dir: Path, eye_name: str) -> OptimizationResult:
    """
    Complete eye tracking pipeline.

    Args:
        csv_path: Path to DLC CSV file
        output_dir: Directory for output files

    Returns:
        Optimization result
    """
    print(f"\n{'='*80}")
    print("EYE TRACKING PIPELINE")
    print(f"{'='*80}")
    print(f"Input:  {csv_path}")
    print(f"Output: {output_dir}\n")

    # 1. Load data
    print("Step 1: Loading data...")
    data = EyeTrackingData.load_from_dlc_csv(filepath=csv_path, min_confidence=0.3)
    print(f"  Loaded {data.n_frames} frames")

    # 2. Filter bad frames
    print("\nStep 2: Filtering frames...")
    data = data.filter_bad_frames(
        min_pupil_points=6,
        require_tear_duct=True
    )

    # 3. Interpolate missing points
    print("\nStep 3: Interpolating missing points...")
    data = data.interpolate_missing_pupil_points()
    print(f"  Interpolation complete")

    # 4. Optimize
    print("\nStep 4: Running bundle adjustment...")

    config = OptimizationConfig(
        max_iterations=500,
        use_huber_loss=True,
        huber_delta_px=2.0,
        pupil_weight=1.0,
        tear_duct_weight=1.0,
        rotation_smoothness_weight=10.0,
        scale_smoothness_weight=5.0,
    )

    initial_guess = EyeModel.create_initial_guess()

    result = optimize_eye_tracking_data(
        observed_pupil_points_px=data.pupil_points_px,
        observed_tear_ducts_px=data.tear_ducts_px,
        camera=PUPIL_LABS_CAMERA,
        initial_eye_model=initial_guess,
        config=config
    )

    # 5. Report results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.num_iterations}")
    print(f"Final cost: {result.final_cost:.6f}")
    print(f"\nReprojection errors:")
    print(f"  Pupil center: {result.mean_pupil_error_px:.2f} ± {result.pupil_center_errors_px.std():.2f} px")
    print(f"  Tear duct:    {result.mean_tear_duct_error_px:.2f} ± {result.tear_duct_errors_px.std():.2f} px")
    print(f"\nPupil dilation:")
    print(f"  Scale: {result.pupil_dialations.mean():.3f} ± {result.pupil_dialations.std():.3f}")
    print(f"  Range: [{result.pupil_dialations.min():.3f}, {result.pupil_dialations.max():.3f}]")
    print(f"\nGaze angles (degrees):")
    azimuth = np.arctan2(result.gaze_directions[:, 0], result.gaze_directions[:, 2])
    elevation = np.arcsin(result.gaze_directions[:, 1])
    print(f"  Azimuth:   {np.rad2deg(azimuth.mean()):.1f} ± {np.rad2deg(azimuth.std()):.1f}°")
    print(f"  Elevation: {np.rad2deg(elevation.mean()):.1f} ± {np.rad2deg(elevation.std()):.1f}°")

    # 6. Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results (model + arrays)
    # TODO: skip this, save results dataframe instead
    # save_full_results(result=result, output_dir=output_dir)

    # Save detailed CSV using Pydantic model
    tracking_results = EyeTrackingResults(
        frame_indices=data.frame_indices,
        pupil_centers_observed_px=data.pupil_points_px.mean(axis=1),
        pupil_centers_reprojected_px=result.projected_pupil_centers_px,
        pupil_centers_3d_mm=result.pupil_centers_3d_mm,
        eyeball_centers_mm=np.tile(
            result.eye_model.eyeball_center_mm,
            (len(data.frame_indices), 1)
        ),
        gaze_directions=result.gaze_directions,
        reprojection_errors_px=result.pupil_center_errors_px
    )

    csv_out = output_dir / f"{eye_name}_tracking_results.csv"
    tracking_results.save_to_csv(filepath=csv_out)
    print(f"\n✓ Saved: {csv_out}")

    return result


if __name__ == "__main__":
    eye0_csv_path = Path("/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/dlc_output/eye_model_v2_model_outputs_iteration_0_flipped/eye0_clipped_4354_11523DLC_Resnet50_eye_model_v2_shuffle1_snapshot_020.csv")
    eye1_csv_path = Path("/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/dlc_output/eye_model_v2_model_outputs_iteration_0_flipped/eye1_clipped_4371_11541_flippedDLC_Resnet50_eye_model_v2_shuffle1_snapshot_020.csv")
    output_dir = Path("output/eye_tracking")

    run_eye_tracking(csv_path=eye0_csv_path, output_dir=output_dir, eye_name="eye0")
    run_eye_tracking(csv_path=eye1_csv_path, output_dir=output_dir, eye_name="eye1")