"""Parallel chunked optimization with soft constraints support."""

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
    data: np.ndarray


@dataclass
class ChunkResult:
    """Result from optimizing a single chunk."""
    chunk_id: int
    global_start: int
    global_end: int
    rotations: np.ndarray
    translations: np.ndarray
    reconstructed: np.ndarray
    reference_geometry: np.ndarray
    computation_time: float
    success: bool


def optimize_single_chunk(
    chunk_task: ChunkTask,
    *,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    optimization_config: 'OptimizationConfig',
    optimize_fn: Callable,
    soft_edges: list[tuple[int, int]] | None = None,
    soft_distances: np.ndarray | None = None,
    lambda_soft: float = 10.0
) -> ChunkResult:
    """Optimize a single chunk with soft constraints."""
    start_time = time.time()

    # Suppress verbose logging in workers
    logging.getLogger('python_code.rigid_body_tracker.core.optimization').setLevel(logging.WARNING)

    try:
        result = optimize_fn(
            noisy_data=chunk_task.data,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config,
            soft_edges=soft_edges,
            soft_distances=soft_distances,
            lambda_soft=lambda_soft
        )

        computation_time = time.time() - start_time

        return ChunkResult(
            chunk_id=chunk_task.chunk_id,
            global_start=chunk_task.global_start,
            global_end=chunk_task.global_end,
            rotations=result.rotations,
            translations=result.translations,
            reconstructed=result.reconstructed,
            reference_geometry=result.reference_geometry,
            computation_time=computation_time,
            success=True
        )

    except Exception as e:
        logger.error(f"Chunk {chunk_task.chunk_id} failed: {e}")

        n_frames = chunk_task.data.shape[0]
        n_markers = chunk_task.data.shape[1]

        return ChunkResult(
            chunk_id=chunk_task.chunk_id,
            global_start=chunk_task.global_start,
            global_end=chunk_task.global_end,
            rotations=np.eye(3)[np.newaxis].repeat(n_frames, axis=0),
            translations=np.zeros((n_frames, 3)),
            reconstructed=chunk_task.data.copy(),
            reference_geometry=np.zeros((n_markers, 3)),
            computation_time=time.time() - start_time,
            success=False
        )


def optimize_chunked_parallel(
    *,
    noisy_data: np.ndarray,
    rigid_edges: list[tuple[int, int]],
    reference_distances: np.ndarray,
    optimization_config: 'OptimizationConfig',
    chunk_config: ChunkConfig,
    optimize_fn: Callable,
    n_workers: int | None = None,
    soft_edges: list[tuple[int, int]] | None = None,
    soft_distances: np.ndarray | None = None,
    lambda_soft: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize long recording using parallel processing with soft constraints.

    Args:
        noisy_data: (n_frames, n_markers, 3)
        rigid_edges: List of rigid edge pairs
        reference_distances: (n_markers, n_markers) initial distance estimates
        optimization_config: OptimizationConfig
        chunk_config: ChunkConfig
        optimize_fn: Optimization function
        n_workers: Number of parallel workers
        soft_edges: Optional soft (flexible) edges
        soft_distances: Optional soft edge distances
        lambda_soft: Weight for soft constraints

    Returns:
        Tuple of (rotations, translations, reconstructed)
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

    if soft_edges:
        logger.info(f"Soft edges: {len(soft_edges)} (flexible constraints)")
        logger.info(f"Soft weight: {lambda_soft}")

    # Check if chunking needed
    if n_frames <= chunk_config.chunk_size + chunk_config.min_chunk_size:
        logger.info("\nData small enough - optimizing as single chunk")
        result = optimize_fn(
            noisy_data=noisy_data,
            rigid_edges=rigid_edges,
            reference_distances=reference_distances,
            config=optimization_config,
            soft_edges=soft_edges,
            soft_distances=soft_distances,
            lambda_soft=lambda_soft
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

    # Create worker function with soft constraints
    worker_fn = partial(
        optimize_single_chunk,
        rigid_edges=rigid_edges,
        reference_distances=reference_distances,
        optimization_config=optimization_config,
        optimize_fn=optimize_fn,
        soft_edges=soft_edges,
        soft_distances=soft_distances,
        lambda_soft=lambda_soft
    )

    # Process chunks in parallel
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING {len(tasks)} CHUNKS IN PARALLEL")
    logger.info(f"{'='*80}\n")

    start_time = time.time()
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        chunk_results_unsorted = []

        for result in pool.imap_unordered(worker_fn, tasks):
            completed += 1
            chunk_results_unsorted.append(result)

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

    # Sort results
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

    # Allocate output arrays
    all_rotations = np.zeros((n_frames, 3, 3))
    all_translations = np.zeros((n_frames, 3))
    all_reconstructed = np.zeros((n_frames, n_markers, 3))

    # Use the first chunk's reference geometry for reconstruction
    reference_geometry = chunk_results[0].reference_geometry

    # Stitch chunks with blending
    logger.info(f"\n{'='*80}")
    logger.info("STITCHING CHUNKS")
    logger.info(f"{'='*80}")

    for chunk_idx, chunk_result in enumerate(chunk_results):
        global_start = chunk_result.global_start
        global_end = chunk_result.global_end

        if chunk_idx == 0:
            # First chunk: copy directly
            blend_end = global_end - chunk_config.overlap_size
            all_rotations[global_start:blend_end] = chunk_result.rotations[:blend_end - global_start]
            all_translations[global_start:blend_end] = chunk_result.translations[:blend_end - global_start]
            all_reconstructed[global_start:blend_end] = chunk_result.reconstructed[:blend_end - global_start]

            logger.info(f"Chunk 0: Copied frames {global_start}-{blend_end}")

        else:
            # Subsequent chunks: blend overlap
            prev_result = chunk_results[chunk_idx - 1]
            overlap_start = global_start
            overlap_end = min(global_start + chunk_config.overlap_size, global_end)
            blend_size = min(chunk_config.blend_window, overlap_end - overlap_start)

            blend_global_start = overlap_start
            blend_global_end = overlap_start + blend_size

            prev_local_start = blend_global_start - prev_result.global_start
            prev_local_end = blend_global_end - prev_result.global_start

            curr_local_start = blend_global_start - global_start
            curr_local_end = blend_global_end - global_start

            R_prev = prev_result.rotations[prev_local_start:prev_local_end]
            T_prev = prev_result.translations[prev_local_start:prev_local_end]

            R_curr = chunk_result.rotations[curr_local_start:curr_local_end]
            T_curr = chunk_result.translations[curr_local_start:curr_local_end]

            weights = create_blend_weights(n_frames=blend_size, blend_type="cosine")

            R_blended = blend_rotations(R1=R_prev, R2=R_curr, weights=weights)
            T_blended = blend_translations(T1=T_prev, T2=T_curr, weights=weights)

            # Reconstruct from blended poses
            recon_blended = np.zeros((blend_size, n_markers, 3))
            for i in range(blend_size):
                recon_blended[i] = (R_blended[i] @ reference_geometry.T).T + T_blended[i]

            all_rotations[blend_global_start:blend_global_end] = R_blended
            all_translations[blend_global_start:blend_global_end] = T_blended
            all_reconstructed[blend_global_start:blend_global_end] = recon_blended

            logger.info(f"Chunk {chunk_idx}: Blended frames {blend_global_start}-{blend_global_end}")

            # Copy non-overlapping region
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
    seconds_per_chunk: float = 750.0
) -> dict[str, float]:
    """Estimate speedup from parallel processing."""
    n_chunks = int(np.ceil(n_frames / chunk_size))
    sequential_time = n_chunks * seconds_per_chunk
    parallel_time = (n_chunks / n_workers) * seconds_per_chunk * 1.1
    speedup = sequential_time / parallel_time

    return {
        'n_chunks': n_chunks,
        'sequential_time_minutes': sequential_time / 60,
        'parallel_time_minutes': parallel_time / 60,
        'speedup': speedup,
        'time_saved_minutes': (sequential_time - parallel_time) / 60
    }