"""Save and load optimized eye model parameters."""

import numpy as np
from pathlib import Path
from pydantic import BaseModel
from numpydantic import NDArray, Shape

from python_code.pyceres_solvers.eye_tracking.eye_pyceres_bundle_adjustment import EyeModel
from python_code.pyceres_solvers.rigid_body_solver.core.optimization import OptimizationResult


class SavedOptimizationResults(BaseModel):
    """Container for saved optimization results."""

    model_config = {"arbitrary_types_allowed": True}

    eye_model: EyeModel
    quaternions: NDArray[Shape["*, 4"], float]
    pupil_scales: NDArray[Shape["* frame_number"], float]
    gaze_directions: NDArray[Shape["*, 3"], float]

    @classmethod
    def from_optimization_result(cls, *, result: OptimizationResult) -> "SavedOptimizationResults":
        """Create from OptimizationResult."""
        return cls(
            eye_model=result.eye_model,
            quaternions=result.eye_quaternions,
            pupil_scales=result.pupil_dialations,
            gaze_directions=result.gaze_directions
        )



def save_full_results(*, result: OptimizationResult, output_dir: Path) -> None:
    """
    Save complete optimization results.
    
    Saves:
    - eye_model.json: Optimized static parameters (using Pydantic)
    - quaternions.npy: Per-frame rotations
    - pupil_scales.npy: Per-frame dilation
    - gaze_directions.npy: Per-frame gaze vectors
    
    Args:
        result: Optimization result
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save eye model using Pydantic serialization
    eye_model_save_path = output_dir / "eye_model.json"
    eye_model_save_path.write_text(result.eye_model.model_dump_json(indent=4))

    
    # Save per-frame arrays
    np.save(file=output_dir / "quaternions.npy", arr=result.eye_quaternions)
    np.save(file=output_dir / "pupil_scales.npy", arr=result.pupil_dialations)
    np.save(file=output_dir / "gaze_directions.npy", arr=result.gaze_directions)
    
    print(f"✓ Saved all results to: {output_dir}")


def load_full_results(*, input_dir: Path) -> SavedOptimizationResults:
    """
    Load complete optimization results.
    
    Args:
        input_dir: Input directory
        
    Returns:
        Loaded optimization results
    """
    eye_mode_path = input_dir / "eye_model.json"
    eye_model = EyeModel.model_validate_json(eye_mode_path.read_text())
    quaternions = np.load(file=input_dir / "quaternions.npy")
    pupil_scales = np.load(file=input_dir / "pupil_scales.npy")
    gaze_directions = np.load(file=input_dir / "gaze_directions.npy")
    
    return SavedOptimizationResults(
        eye_model=eye_model,
        quaternions=quaternions,
        pupil_scales=pupil_scales,
        gaze_directions=gaze_directions
    )
