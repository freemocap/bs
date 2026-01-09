"""Chunked optimization for long recordings with accuracy preservation."""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Callable
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunked optimization."""
    
    chunk_size: int = 500
    """Number of frames per chunk"""
    
    overlap_size: int = 50
    """Number of overlapping frames between chunks"""
    
    blend_window: int = 25
    """Size of blending window in overlap region"""
    
    min_chunk_size: int = 100
    """Minimum frames to process as separate chunk"""


def split_into_chunks(
    *,
    n_frames: int,
    chunk_size: int,
    overlap_size: int
) -> list[tuple[int, int, int, int]]:
    """
    Split frame range into overlapping chunks.
    
    Args:
        n_frames: Total number of frames
        chunk_size: Frames per chunk
        overlap_size: Overlapping frames between chunks
        
    Returns:
        List of (global_start, global_end, local_blend_start, local_blend_end) tuples
        - global_start/end: indices in original array
        - local_blend_start/end: where to blend in this chunk's local coordinates
    """
    chunks = []
    stride = chunk_size - overlap_size
    
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        
        # Determine blend region in local coordinates
        if len(chunks) == 0:
            # First chunk: no blending at start
            local_blend_start = 0
        else:
            # Blend region is at the start of this chunk
            local_blend_start = 0
            
        if end == n_frames:
            # Last chunk: no blending at end
            local_blend_end = end - start
        else:
            # Blend region is at the end of this chunk
            local_blend_end = overlap_size
            
        chunks.append((start, end, local_blend_start, local_blend_end))
        
        if end == n_frames:
            break
            
        start += stride
    
    return chunks


def blend_rotations(
    *,
    R1: np.ndarray,
    R2: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Blend rotation matrices using spherical interpolation (SLERP).
    
    Args:
        R1: (n_frames, 3, 3) rotation matrices from chunk 1
        R2: (n_frames, 3, 3) rotation matrices from chunk 2
        weights: (n_frames,) blend weights (0=R1, 1=R2)
        
    Returns:
        (n_frames, 3, 3) blended rotations
    """
    n_frames = len(weights)
    blended = np.zeros((n_frames, 3, 3))
    
    for i in range(n_frames):
        if weights[i] <= 0.0:
            blended[i] = R1[i]
        elif weights[i] >= 1.0:
            blended[i] = R2[i]
        else:
            # Convert to quaternions
            q1 = Rotation.from_matrix(R1[i]).as_quat()
            q2 = Rotation.from_matrix(R2[i]).as_quat()
            
            # Ensure same hemisphere
            if np.dot(q1, q2) < 0:
                q2 = -q2
            
            # SLERP
            rot_interp = Slerp(
                times=[0, 1],
                rotations=Rotation.from_quat([q1, q2])
            )
            blended[i] = rot_interp(weights[i]).as_matrix()
    
    return blended


def blend_translations(
    *,
    T1: np.ndarray,
    T2: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Blend translations using linear interpolation.
    
    Args:
        T1: (n_frames, 3) translations from chunk 1
        T2: (n_frames, 3) translations from chunk 2
        weights: (n_frames,) blend weights (0=T1, 1=T2)
        
    Returns:
        (n_frames, 3) blended translations
    """
    weights_expanded = weights[:, np.newaxis]
    return (1 - weights_expanded) * T1 + weights_expanded * T2


def create_blend_weights(*, n_frames: int, blend_type: str = "cosine") -> np.ndarray:
    """
    Create smooth blending weights.
    
    Args:
        n_frames: Length of blend region
        blend_type: "linear" or "cosine"
        
    Returns:
        (n_frames,) weights from 0 to 1
    """
    if blend_type == "linear":
        return np.linspace(0, 1, n_frames)
    elif blend_type == "cosine":
        # Cosine interpolation for smoother blending
        t = np.linspace(0, 1, n_frames)
        return (1 - np.cos(t * np.pi)) / 2
    else:
        raise ValueError(f"Unknown blend_type: {blend_type}")


def optimize_chunk_with_initialization(
    *,
    original_data: np.ndarray,
    reference_geometry: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    config: 'OptimizationConfig',
    optimize_fn: Callable,
    initial_poses: list[tuple[np.ndarray, np.ndarray]] | None = None
) -> 'OptimizationResult':
    """
    Optimize a chunk with optional initialization from previous chunk.
    
    Args:
        original_data: (n_frames, n_markers, 3) for this chunk
        reference_geometry: (n_markers, 3)
        rigid_edges: List of rigid edge pairs
        reference_distances: (n_markers, n_markers) distance matrix
        config: OptimizationConfig
        optimize_fn: Function that performs optimization
        initial_poses: Optional list of (quat, trans) to initialize first frames
        
    Returns:
        OptimizationResult
    """
    # If we have initial poses from previous chunk, we could use them
    # For now, just run standard optimization
    # (In production, you'd modify optimize_rigid_body to accept initial_poses)
    
    return optimize_fn(
        original_data=original_data,
        reference_geometry=reference_geometry,
        rigid_edges=rigid_edges,
        reference_distances=reference_distances,
        config=config
    )


def optimize_chunked(
    *,
    original_data: np.ndarray,
    reference_geometry: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    optimization_config: 'OptimizationConfig',
    chunk_config: ChunkConfig,
    optimize_fn: Callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize long recording using overlapping chunks.
    
    This approach:
    1. Splits data into overlapping chunks
    2. Optimizes each chunk independently (can be parallelized!)
    3. Blends overlapping regions smoothly using SLERP for rotations
    4. Returns full-length optimized trajectory
    
    Args:
        original_data: (n_frames, n_markers, 3) full original data
        reference_geometry: (n_markers, 3) reference shape
        rigid_edges: List of rigid edge pairs
        reference_distances: Distance matrix
        optimization_config: Config for optimization
        chunk_config: Config for chunking
        optimize_fn: Optimization function (e.g., optimize_rigid_body)
        
    Returns:
        Tuple of:
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - reconstructed: (n_frames, n_markers, 3)
    """
    n_frames, n_markers, _ = original_data.shape
    
    logger.info("="*80)
    logger.info("CHUNKED OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Total frames: {n_frames}")
    logger.info(f"Chunk size: {chunk_config.chunk_size}")
    logger.info(f"Overlap: {chunk_config.overlap_size}")
    logger.info(f"Blend window: {chunk_config.blend_window}")
    
    # Check if chunking is needed
    if n_frames <= chunk_config.chunk_size + chunk_config.min_chunk_size:
        logger.info("\nData small enough - optimizing as single chunk")
        result = optimize_fn(
            original_data=original_data,
            reference_geometry=reference_geometry,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config
        )
        return result.rotations, result.translations, result.reconstructed_keypoints
    
    # Split into chunks
    chunks = split_into_chunks(
        n_frames=n_frames,
        chunk_size=chunk_config.chunk_size,
        overlap_size=chunk_config.overlap_size
    )
    
    logger.info(f"\nSplit into {len(chunks)} chunks:")
    for i, (start, end, _, _) in enumerate(chunks):
        logger.info(f"  Chunk {i}: frames {start}-{end} ({end-start} frames)")
    
    # Allocate output arrays
    all_rotations = np.zeros((n_frames, 3, 3))
    all_translations = np.zeros((n_frames, 3))
    all_reconstructed = np.zeros((n_frames, n_markers, 3))
    
    # Process each chunk
    chunk_results = []
    
    for chunk_idx, (global_start, global_end, local_blend_start, local_blend_end) in enumerate(chunks):
        chunk_frames = global_end - global_start
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING CHUNK {chunk_idx + 1}/{len(chunks)}")
        logger.info(f"{'='*80}")
        logger.info(f"Global range: [{global_start}, {global_end})")
        logger.info(f"Chunk frames: {chunk_frames}")
        
        # Extract chunk data
        chunk_data = original_data[global_start:global_end]
        
        # Optimize this chunk
        result = optimize_fn(
            original_data=chunk_data,
            reference_geometry=reference_geometry,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config
        )
        
        chunk_results.append({
            'global_start': global_start,
            'global_end': global_end,
            'rotations': result.rotations,
            'translations': result.translations,
            'reconstructed': result.reconstructed_keypoints
        })
    
    # Stitch chunks together with blending
    logger.info(f"\n{'='*80}")
    logger.info("STITCHING CHUNKS")
    logger.info(f"{'='*80}")
    
    for chunk_idx, chunk_result in enumerate(chunk_results):
        global_start = chunk_result['global_start']
        global_end = chunk_result['global_end']
        
        if chunk_idx == 0:
            # First chunk: copy directly (no previous chunk to blend with)
            blend_end = global_end - chunk_config.overlap_size
            all_rotations[global_start:blend_end] = chunk_result['rotations'][:blend_end - global_start]
            all_translations[global_start:blend_end] = chunk_result['translations'][:blend_end - global_start]
            all_reconstructed[global_start:blend_end] = chunk_result['reconstructed'][:blend_end - global_start]
            
            logger.info(f"Chunk 0: Copied frames {global_start}-{blend_end}")
            
        else:
            # Subsequent chunks: blend overlap region with previous chunk
            prev_result = chunk_results[chunk_idx - 1]
            overlap_start = global_start
            overlap_end = min(global_start + chunk_config.overlap_size, global_end)
            blend_size = min(chunk_config.blend_window, overlap_end - overlap_start)
            
            # Blend region: [overlap_start, overlap_start + blend_size)
            blend_global_start = overlap_start
            blend_global_end = overlap_start + blend_size
            
            # Extract data from both chunks in the blend region
            prev_local_start = blend_global_start - prev_result['global_start']
            prev_local_end = blend_global_end - prev_result['global_start']
            
            curr_local_start = blend_global_start - global_start
            curr_local_end = blend_global_end - global_start
            
            R_prev = prev_result['rotations'][prev_local_start:prev_local_end]
            T_prev = prev_result['translations'][prev_local_start:prev_local_end]
            
            R_curr = chunk_result['rotations'][curr_local_start:curr_local_end]
            T_curr = chunk_result['translations'][curr_local_start:curr_local_end]
            
            # Create blend weights
            weights = create_blend_weights(n_frames=blend_size, blend_type="cosine")
            
            # Blend rotations and translations
            R_blended = blend_rotations(R1=R_prev, R2=R_curr, weights=weights)
            T_blended = blend_translations(T1=T_prev, T2=T_curr, weights=weights)
            
            # Reconstruct points from blended poses
            recon_blended = np.zeros((blend_size, n_markers, 3))
            for i in range(blend_size):
                recon_blended[i] = (R_blended[i] @ reference_geometry.T).T + T_blended[i]
            
            # Store blended region
            all_rotations[blend_global_start:blend_global_end] = R_blended
            all_translations[blend_global_start:blend_global_end] = T_blended
            all_reconstructed[blend_global_start:blend_global_end] = recon_blended
            
            logger.info(f"Chunk {chunk_idx}: Blended frames {blend_global_start}-{blend_global_end}")
            
            # Copy non-overlapping region from current chunk
            copy_start = blend_global_end
            copy_end = global_end - (chunk_config.overlap_size if chunk_idx < len(chunk_results) - 1 else 0)
            
            if copy_start < copy_end:
                local_copy_start = copy_start - global_start
                local_copy_end = copy_end - global_start
                
                all_rotations[copy_start:copy_end] = chunk_result['rotations'][local_copy_start:local_copy_end]
                all_translations[copy_start:copy_end] = chunk_result['translations'][local_copy_start:local_copy_end]
                all_reconstructed[copy_start:copy_end] = chunk_result['reconstructed'][local_copy_start:local_copy_end]
                
                logger.info(f"Chunk {chunk_idx}: Copied frames {copy_start}-{copy_end}")
    
    logger.info(f"\n✓ Stitching complete: {n_frames} frames")
    
    return all_rotations, all_translations, all_reconstructed
