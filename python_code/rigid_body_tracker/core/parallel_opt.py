"""Parallel chunked optimization for maximum speed."""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Callable
import multiprocessing as mp
from functools import partial
import time

from python_code.rigid_body_tracker.core.chunking import (
    ChunkConfig,
    split_into_chunks,
    blend_rotations,
    blend_translations,
    create_blend_weights
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkTask:
    """Task definition for a single chunk."""
    chunk_id: int
    global_start: int
    global_end: int
    data: np.ndarray  # (n_frames_in_chunk, n_markers, 3)


@dataclass
class ChunkResult:
    """Result from optimizing a single chunk."""
    chunk_id: int
    global_start: int
    global_end: int
    rotations: np.ndarray  # (n_frames_in_chunk, 3, 3)
    translations: np.ndarray  # (n_frames_in_chunk, 3)
    reconstructed: np.ndarray  # (n_frames_in_chunk, n_markers, 3)
    computation_time: float
    success: bool


def optimize_single_chunk(
    chunk_task: ChunkTask,
    *,
    reference_geometry: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    optimization_config: 'OptimizationConfig',
    optimize_fn: Callable
) -> ChunkResult:
    """
    Optimize a single chunk (worker function for multiprocessing).

    This function is designed to be called in a separate process.
    """
    start_time = time.time()

    # Suppress verbose logging in worker processes
    logging.getLogger('python_code.rigid_body_tracker.core.optimization').setLevel(logging.WARNING)

    try:
        result = optimize_fn(
            noisy_data=chunk_task.data,
            reference_geometry=reference_geometry,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config
        )

        computation_time = time.time() - start_time

        return ChunkResult(
            chunk_id=chunk_task.chunk_id,
            global_start=chunk_task.global_start,
            global_end=chunk_task.global_end,
            rotations=result.rotations,
            translations=result.translations,
            reconstructed=result.reconstructed,
            computation_time=computation_time,
            success=True
        )

    except Exception as e:
        logger.error(f"Chunk {chunk_task.chunk_id} failed: {e}")

        # Return dummy result
        n_frames = chunk_task.data.shape[0]
        n_markers = chunk_task.data.shape[1]

        return ChunkResult(
            chunk_id=chunk_task.chunk_id,
            global_start=chunk_task.global_start,
            global_end=chunk_task.global_end,
            rotations=np.eye(3)[np.newaxis].repeat(n_frames, axis=0),
            translations=np.zeros((n_frames, 3)),
            reconstructed=chunk_task.data.copy(),
            computation_time=time.time() - start_time,
            success=False
        )


def optimize_chunked_parallel(
    *,
    noisy_data: np.ndarray,
    reference_geometry: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    optimization_config: 'OptimizationConfig',
    chunk_config: ChunkConfig,
    optimize_fn: Callable,
    n_workers: int | None = None,
    show_progress: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize long recording using parallel processing of chunks.

    This is the FAST version - optimizes multiple chunks simultaneously!

    Args:
        noisy_data: (n_frames, n_markers, 3) full noisy data
        reference_geometry: (n_markers, 3) reference shape
        rigid_edges: List of rigid edge pairs
        reference_distances: Distance matrix
        optimization_config: Config for optimization
        chunk_config: Config for chunking
        optimize_fn: Optimization function (e.g., optimize_rigid_body)
        n_workers: Number of parallel workers (None = use all CPU cores)
        show_progress: Whether to show progress updates

    Returns:
        Tuple of:
        - rotations: (n_frames, 3, 3)
        - translations: (n_frames, 3)
        - reconstructed: (n_frames, n_markers, 3)
    """
    n_frames, n_markers, _ = noisy_data.shape

    if n_workers is None:
        n_workers = mp.cpu_count()

    logger.info("="*80)
    logger.info("PARALLEL CHUNKED OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Total frames: {n_frames}")
    logger.info(f"Chunk size: {chunk_config.chunk_size}")
    logger.info(f"Overlap: {chunk_config.overlap_size}")
    logger.info(f"Workers: {n_workers}")

    # Check if chunking is needed
    if n_frames <= chunk_config.chunk_size + chunk_config.min_chunk_size:
        logger.info("\nData small enough - optimizing as single chunk")
        result = optimize_fn(
            noisy_data=noisy_data,
            reference_geometry=reference_geometry,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config
        )
        return result.rotations, result.translations, result.reconstructed

    # Split into chunks
    chunks = split_into_chunks(
        n_frames=n_frames,
        chunk_size=chunk_config.chunk_size,
        overlap_size=chunk_config.overlap_size
    )

    logger.info(f"\nSplit into {len(chunks)} chunks:")
    for i, (start, end, _, _) in enumerate(chunks):
        logger.info(f"  Chunk {i}: frames {start}-{end} ({end-start} frames)")

    # Create tasks
    tasks = []
    for chunk_id, (global_start, global_end, _, _) in enumerate(chunks):
        chunk_data = noisy_data[global_start:global_end]
        tasks.append(ChunkTask(
            chunk_id=chunk_id,
            global_start=global_start,
            global_end=global_end,
            data=chunk_data
        ))

    # Create worker function with fixed arguments
    worker_fn = partial(
        optimize_single_chunk,
        reference_geometry=reference_geometry,
        rigid_edges=rigid_edges,
        reference_distances=reference_distances,
        optimization_config=optimization_config,
        optimize_fn=optimize_fn
    )

    # Process chunks in parallel
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING {len(tasks)} CHUNKS IN PARALLEL")
    logger.info(f"{'='*80}\n")

    start_time = time.time()
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        # Use imap_unordered for progress updates
        chunk_results_unsorted = []

        for result in pool.imap_unordered(worker_fn, tasks):
            completed += 1
            chunk_results_unsorted.append(result)

            if show_progress:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (len(tasks) - completed) * avg_time

                logger.info(
                    f"✓ Chunk {result.chunk_id} complete "
                    f"({completed}/{len(tasks)}) - "
                    f"{result.computation_time:.1f}s - "
                    f"ETA: {remaining/60:.1f}min"
                )

    total_time = time.time() - start_time

    # Sort results by chunk_id
    chunk_results = sorted(chunk_results_unsorted, key=lambda x: x.chunk_id)

    # Check for failures
    failed_chunks = [r.chunk_id for r in chunk_results if not r.success]
    if failed_chunks:
        logger.error(f"Failed chunks: {failed_chunks}")
        raise RuntimeError(f"Optimization failed for chunks: {failed_chunks}")

    logger.info(f"\n{'='*80}")
    logger.info("PARALLEL OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average time per chunk: {total_time/len(tasks):.1f}s")
    logger.info(f"Speedup vs sequential: ~{len(tasks) * (total_time/len(tasks)) / total_time:.1f}x")

    # Allocate output arrays
    all_rotations = np.zeros((n_frames, 3, 3))
    all_translations = np.zeros((n_frames, 3))
    all_reconstructed = np.zeros((n_frames, n_markers, 3))

    # Stitch chunks together with blending
    logger.info(f"\n{'='*80}")
    logger.info("STITCHING CHUNKS")
    logger.info(f"{'='*80}")

    for chunk_idx, chunk_result in enumerate(chunk_results):
        global_start = chunk_result.global_start
        global_end = chunk_result.global_end

        if chunk_idx == 0:
            # First chunk: copy directly (no previous chunk to blend with)
            blend_end = global_end - chunk_config.overlap_size
            all_rotations[global_start:blend_end] = chunk_result.rotations[:blend_end - global_start]
            all_translations[global_start:blend_end] = chunk_result.translations[:blend_end - global_start]
            all_reconstructed[global_start:blend_end] = chunk_result.reconstructed[:blend_end - global_start]

            logger.info(f"Chunk 0: Copied frames {global_start}-{blend_end}")

        else:
            # Subsequent chunks: blend overlap region with previous chunk
            prev_result = chunk_results[chunk_idx - 1]
            overlap_start = global_start
            overlap_end = min(global_start + chunk_config.overlap_size, global_end)
            blend_size = min(chunk_config.blend_window, overlap_end - overlap_start)

            # Blend region
            blend_global_start = overlap_start
            blend_global_end = overlap_start + blend_size

            # Extract data from both chunks
            prev_local_start = blend_global_start - prev_result.global_start
            prev_local_end = blend_global_end - prev_result.global_start

            curr_local_start = blend_global_start - global_start
            curr_local_end = blend_global_end - global_start

            R_prev = prev_result.rotations[prev_local_start:prev_local_end]
            T_prev = prev_result.translations[prev_local_start:prev_local_end]

            R_curr = chunk_result.rotations[curr_local_start:curr_local_end]
            T_curr = chunk_result.translations[curr_local_start:curr_local_end]

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

                all_rotations[copy_start:copy_end] = chunk_result.rotations[local_copy_start:local_copy_end]
                all_translations[copy_start:copy_end] = chunk_result.translations[local_copy_start:local_copy_end]
                all_reconstructed[copy_start:copy_end] = chunk_result.reconstructed[local_copy_start:local_copy_end]

                logger.info(f"Chunk {chunk_idx}: Copied frames {copy_start}-{copy_end}")

    logger.info(f"\n✓ Stitching complete: {n_frames} frames")

    return all_rotations, all_translations, all_reconstructed


def estimate_parallel_speedup(
    *,
    n_frames: int,
    chunk_size: int,
    n_workers: int,
    seconds_per_chunk: float = 750.0  # ~12.5 minutes
) -> dict[str, float]:
    """
    Estimate speedup from parallel processing.

    Args:
        n_frames: Total number of frames
        chunk_size: Frames per chunk
        n_workers: Number of parallel workers
        seconds_per_chunk: Estimated time to process one chunk

    Returns:
        Dictionary with timing estimates
    """
    n_chunks = int(np.ceil(n_frames / chunk_size))

    # Sequential time
    sequential_time = n_chunks * seconds_per_chunk

    # Parallel time (assuming perfect scaling, slight overhead)
    parallel_time = (n_chunks / n_workers) * seconds_per_chunk * 1.1  # 10% overhead

    speedup = sequential_time / parallel_time

    return {
        'n_chunks': n_chunks,
        'sequential_time_minutes': sequential_time / 60,
        'parallel_time_minutes': parallel_time / 60,
        'speedup': speedup,
        'time_saved_minutes': (sequential_time - parallel_time) / 60
    }