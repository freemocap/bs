"""Active contour (snake) fitting for pupil edge detection.

Uses initial ellipse fit as starting point and deforms to find pupil boundary.
"""

import logging
import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SnakeParams(BaseModel):
    """Parameters for active contour snake fitting."""
    n_points: int = 20
    alpha: float = 0.5  # Snake length shape parameter (HIGH = resists length changes)
    beta: float = 1.0  # Snake smoothness shape parameter (HIGH = stays smooth)
    w_line: float = 0.0  # Line energy weight
    w_edge: float = 1.0  # Edge energy weight
    gamma: float = 0.01  # Time step
    max_iterations: int = 500  # FEWER iterations = less time to collapse
    sigma: float = 2.0  # Gaussian blur sigma for edge detection
    max_displacement: float = 5.0  # Maximum allowed displacement from initial position (pixels)


class SnakeContour(BaseModel):
    """Result of snake fitting."""
    points: np.ndarray  # (n_points, 2) array of x,y coordinates
    converged: bool
    n_iterations: int

    class Config:
        arbitrary_types_allowed = True


def generate_ellipse_points(
    *,
    ellipse_params: np.ndarray,
    n_points: int
) -> np.ndarray:
    """Generate points along an ellipse.

    Args:
        ellipse_params: [cx, cy, a, b, theta] parameters
        n_points: Number of points to generate

    Returns:
        (n_points, 2) array of x,y coordinates
    """
    cx, cy, semi_major, semi_minor, rotation = ellipse_params

    # Generate angles
    theta = np.linspace(start=0, stop=2*np.pi, num=n_points, endpoint=False)

    # Parametric ellipse in local coordinates
    x_local = semi_major * np.cos(theta)
    y_local = semi_minor * np.sin(theta)

    # Rotate and translate to world coordinates
    cos_t = np.cos(rotation)
    sin_t = np.sin(rotation)

    x = cx + x_local * cos_t - y_local * sin_t
    y = cy + x_local * sin_t + y_local * cos_t

    return np.column_stack([x, y])


def fit_snake_to_pupil(
    *,
    image: np.ndarray,
    initial_ellipse_params: np.ndarray,
    params: SnakeParams | None = None
) -> SnakeContour:
    """Fit active contour snake to pupil edge with displacement constraints.

    The snake starts from the initial ellipse and deforms slightly to find
    the edge of the pupil. High rigidity parameters and displacement
    constraints keep it from collapsing or wandering too far.

    Args:
        image: Input image (BGR format from OpenCV)
        initial_ellipse_params: Initial ellipse [cx, cy, a, b, theta]
        params: Snake parameters (uses defaults if None)

    Returns:
        SnakeContour with fitted points and convergence info

    Raises:
        ValueError: If initial ellipse parameters are invalid
    """
    if params is None:
        params = SnakeParams()

    # Validate ellipse parameters
    if np.isnan(initial_ellipse_params).any():
        raise ValueError("Initial ellipse parameters contain NaN")

    if len(initial_ellipse_params) != 5:
        raise ValueError(f"Expected 5 ellipse parameters, got {len(initial_ellipse_params)}")

    # Convert to grayscale
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise for better edge detection
    if params.sigma > 0:
        gray = gaussian(image=gray, sigma=params.sigma, preserve_range=True).astype(np.uint8)

    # Generate initial snake points from ellipse
    init_snake = generate_ellipse_points(
        ellipse_params=initial_ellipse_params,
        n_points=params.n_points
    )

    # Run active contour with constraints
    # Important: scikit-image uses (row, col) = (y, x) by default
    # We need to convert our (x, y) points to (y, x) for scikit-image
    # Then convert back to (x, y) afterwards
    init_snake_yx = init_snake[:, [1, 0]]  # Swap x,y to y,x

    snake_points_yx = active_contour(
        image=gray,
        snake=init_snake_yx,
        alpha=params.alpha,
        beta=params.beta,
        w_line=params.w_line,
        w_edge=params.w_edge,
        gamma=params.gamma,
        max_num_iter=params.max_iterations,  # Correct parameter name!
    )

    # Convert back from (y, x) to (x, y)
    snake_points = snake_points_yx[:, [1, 0]]
    # Constrain displacement: prevent points from moving too far from initial positions
    # This keeps the snake anchored near the original ellipse-based p1-p8 estimates
    if params.max_displacement > 0:
        displacement = snake_points - init_snake
        distances = np.linalg.norm(displacement, axis=1)

        # For points that moved too far, clamp them to max_displacement radius
        too_far = distances > params.max_displacement
        if np.any(too_far):
            # Scale back the displacement to exactly max_displacement
            scale_factors = params.max_displacement / distances[too_far]
            snake_points[too_far] = init_snake[too_far] + displacement[too_far] * scale_factors[:, np.newaxis]
            logger.debug(f"Clamped {np.sum(too_far)} points that moved beyond {params.max_displacement}px")

    # Check convergence (simple heuristic: if points moved significantly)
    displacement = np.linalg.norm(snake_points - init_snake, axis=1)
    mean_displacement = np.mean(displacement)
    converged = mean_displacement < 5.0  # pixels

    return SnakeContour(
        points=snake_points,
        converged=converged,
        n_iterations=params.max_iterations  # scikit-image doesn't return actual iteration count
    )


def fit_snake_safe(
    *,
    image: np.ndarray,
    initial_ellipse_params: np.ndarray,
    params: SnakeParams | None = None
) -> SnakeContour | None:
    """Safe wrapper for snake fitting that returns None on failure.

    Args:
        image: Input image (BGR format)
        initial_ellipse_params: Initial ellipse parameters
        params: Snake parameters

    Returns:
        SnakeContour if successful, None if fitting fails
    """
    try:
        return fit_snake_to_pupil(
            image=image,
            initial_ellipse_params=initial_ellipse_params,
            params=params
        )
    except (ValueError, RuntimeError, cv2.error) as e:
        logger.warning(f"Snake fitting failed: {e}")
        return None