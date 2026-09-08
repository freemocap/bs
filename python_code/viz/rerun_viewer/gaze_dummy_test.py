"""
Gaze-in-skull-frame diagnostic viewer.

Shows gaze arrows for both eyes with skull translation AND rotation removed,
so the skull reference geometry is stationary and any arrow movement is
purely eye-in-head rotation. Good for confirming the gaze pipeline is
computing eye-relative motion correctly.

What to look for:
  - Gold keypoints/edges: skull reference geometry (should be perfectly still)
  - Blue/orange dots: eye center positions (should sit exactly on the gold
    left_eye / right_eye reference keypoints — confirms world-to-skull
    transform is correct)
  - Blue arrow: left-eye gaze direction in skull frame
  - Orange arrow: right-eye gaze direction in skull frame
  - If arrows move → eye IS rotating relative to skull
  - If they are static → gaze is not changing relative to head
"""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from python_code.ferret_gaze.calculate_gaze.ferret_gaze_kinematics import FerretGazeKinematics
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.rigid_body_solver.viz.ferret_skull_rerun import load_kinematics_from_tidy_csv
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

COLOR_LEFT = (0, 150, 255)
COLOR_RIGHT = (255, 100, 0)
COLOR_SKULL = (200, 200, 80)
COLOR_WORLD_X = (180, 50, 50)
COLOR_WORLD_Y = (50, 180, 50)
COLOR_WORLD_Z = (50, 50, 180)

GAZE_SCALE_MM: float = 80.0


def _to_skull_local(
    R_skull: np.ndarray,        # (N, 3, 3)
    skull_pos: np.ndarray,      # (N, 3)
    world_positions: np.ndarray,  # (N, 3)
    world_vectors: np.ndarray | None = None,  # (N, 3)
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Transform world-space positions (and optionally directions) into skull-local frame.
    positions: local = R^T @ (world - skull_pos)
    vectors:   local = R^T @ world_dir  (no translation)
    """
    R_inv = R_skull.transpose(0, 2, 1)                     # (N, 3, 3)
    local_pos = np.einsum("nij,nj->ni", R_inv, world_positions - skull_pos)
    local_dir = np.einsum("nij,nj->ni", R_inv, world_vectors) if world_vectors is not None else None
    return local_pos, local_dir


def plot_gaze_in_skull_frame(
    recording_folder: RecordingFolder,
    gaze_scale_mm: float = GAZE_SCALE_MM,
) -> None:

    # ── Load skull kinematics ─────────────────────────────────────────────
    if recording_folder.skull_reference_geometry is None:
        raise FileNotFoundError("Skull reference geometry not found.")
    if recording_folder.skull_kinematics_csv is None:
        raise FileNotFoundError("Skull kinematics CSV not found.")

    ref_geo = ReferenceGeometry.from_json_file(recording_folder.skull_reference_geometry)
    skull = load_kinematics_from_tidy_csv(
        csv_path=recording_folder.skull_kinematics_csv,
        reference_geometry=ref_geo,
        name="skull",
    )
    print(f"Skull kinematics: {skull.n_frames} frames")

    # ── Load gaze kinematics ──────────────────────────────────────────────
    if recording_folder.gaze_kinematics is None:
        raise FileNotFoundError("Gaze kinematics directory not found.")

    left_gaze = FerretGazeKinematics.load_from_directory(
        eye_name="left_gaze",
        input_directory=recording_folder.gaze_kinematics,
    )
    right_gaze = FerretGazeKinematics.load_from_directory(
        eye_name="right_gaze",
        input_directory=recording_folder.gaze_kinematics,
    )
    print(f"Left gaze: {left_gaze.n_frames} frames")
    print(f"Right gaze: {right_gaze.n_frames} frames")

    # ── Align to skull timestamps ─────────────────────────────────────────
    skull_ts = skull.timestamps
    t0 = skull_ts[0]
    times = skull_ts - t0
    n = skull.n_frames

    def interp_gaze(gk: FerretGazeKinematics) -> tuple[np.ndarray, np.ndarray]:
        ts = gk.kinematics.timestamps
        pos = np.stack(
            [np.interp(skull_ts, ts, gk.kinematics.position_xyz[:, i]) for i in range(3)],
            axis=-1,
        )
        dirs = np.stack(
            [np.interp(skull_ts, ts, gk.gaze_directions[:, i]) for i in range(3)],
            axis=-1,
        )
        return pos, dirs

    left_eye_world, left_dirs_world = interp_gaze(left_gaze)
    right_eye_world, right_dirs_world = interp_gaze(right_gaze)

    # ── Transform to skull-local frame ───────────────────────────────────
    R_skull = skull.orientations.to_rotation_matrices()  # (N, 3, 3)
    skull_pos = skull.position_xyz                       # (N, 3)

    left_eye_local, left_dirs_local = _to_skull_local(R_skull, skull_pos, left_eye_world, left_dirs_world)
    right_eye_local, right_dirs_local = _to_skull_local(R_skull, skull_pos, right_eye_world, right_dirs_world)

    # ── Sanity-check: eye centers should match reference geometry ─────────
    kp = ref_geo.keypoints
    for label, local_pos in [("left_eye", left_eye_local), ("right_eye", right_eye_local)]:
        if label in kp:
            ref = np.array([kp[label].x, kp[label].y, kp[label].z])
            err = float(np.mean(np.linalg.norm(local_pos - ref, axis=-1)))
            print(f"  {label} center mean error vs. reference: {err:.3f} mm  (should be ~0 if math is correct)")

    # ── Compute arrow vectors (scale directions to mm) ────────────────────
    left_vecs = left_dirs_local * gaze_scale_mm    # (N, 3)
    right_vecs = right_dirs_local * gaze_scale_mm  # (N, 3)

    # ── Per-row color arrays (length N, matching partition) ───────────────
    left_eye_col = np.tile(np.array([*COLOR_LEFT, 200], dtype=np.uint8), (n, 1))   # (N, 4)
    right_eye_col = np.tile(np.array([*COLOR_RIGHT, 200], dtype=np.uint8), (n, 1)) # (N, 4)
    left_arrow_col = np.tile(np.array([*COLOR_LEFT], dtype=np.uint8), (n, 1))      # (N, 3)
    right_arrow_col = np.tile(np.array([*COLOR_RIGHT], dtype=np.uint8), (n, 1))    # (N, 3)

    # ── Init Rerun ────────────────────────────────────────────────────────
    rr.init("gaze_skull_local", spawn=True)

    # ── Static skull reference geometry ──────────────────────────────────
    kp_names = list(kp.keys())
    kp_pos = np.array([[kp[n].x, kp[n].y, kp[n].z] for n in kp_names])

    rr.log(
        "skull_local/reference/keypoints",
        rr.Points3D(
            positions=kp_pos,
            labels=kp_names,
            radii=np.full(len(kp_pos), 5.0),
            colors=np.tile(np.array(COLOR_SKULL, dtype=np.uint8), (len(kp_pos), 1)),
        ),
        static=True,
    )

    edges = list(ref_geo.display_edges) if ref_geo.display_edges else ref_geo.get_rigid_edges()
    strips = [
        np.array([[kp[a].x, kp[a].y, kp[a].z], [kp[b].x, kp[b].y, kp[b].z]])
        for a, b in edges
        if a in kp and b in kp
    ]
    if strips:
        rr.log(
            "skull_local/reference/edges",
            rr.LineStrips3D(
                strips=strips,
                colors=np.tile(np.array(COLOR_SKULL, dtype=np.uint8), (len(strips), 1)),
                radii=np.full(len(strips), 1.5),
            ),
            static=True,
        )

    # ── Skull-frame axes at origin ────────────────────────────────────────
    ax_len = 40.0
    rr.log(
        "skull_local/skull_axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * ax_len,
            colors=np.array([COLOR_WORLD_X, COLOR_WORLD_Y, COLOR_WORLD_Z], dtype=np.uint8),
            radii=np.full(3, 0.6),
        ),
        static=True,
    )

    # ── Static style: set radii for animated entities ─────────────────────
    rr.log("skull_local/left_eye_center", rr.Points3D.from_fields(radii=4.0), static=True)
    rr.log("skull_local/right_eye_center", rr.Points3D.from_fields(radii=4.0), static=True)
    rr.log("skull_local/left_gaze", rr.Arrows3D.from_fields(radii=2.0), static=True)
    rr.log("skull_local/right_gaze", rr.Arrows3D.from_fields(radii=2.0), static=True)

    # ── Animated: eye center dots ─────────────────────────────────────────
    rr.send_columns(
        "skull_local/left_eye_center",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Points3D.columns(
                positions=left_eye_local,
                colors=left_eye_col,
            ).partition(lengths=[1] * n)
        ],
    )
    rr.send_columns(
        "skull_local/right_eye_center",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Points3D.columns(
                positions=right_eye_local,
                colors=right_eye_col,
            ).partition(lengths=[1] * n)
        ],
    )

    # ── Animated: gaze arrows in skull-local frame ────────────────────────
    rr.send_columns(
        "skull_local/left_gaze",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=left_eye_local,
                vectors=left_vecs,
                colors=left_arrow_col,
            ).partition(lengths=[1] * n)
        ],
    )
    rr.send_columns(
        "skull_local/right_gaze",
        indexes=[rr.TimeColumn("time", duration=times)],
        columns=[
            *rr.Arrows3D.columns(
                origins=right_eye_local,
                vectors=right_vecs,
                colors=right_arrow_col,
            ).partition(lengths=[1] * n)
        ],
    )

    # ── Blueprint ─────────────────────────────────────────────────────────
    view = rrb.Spatial3DView(
        name="Gaze in Skull Frame",
        origin="/",
        contents=["+ skull_local/**"],
        eye_controls=rrb.EyeControls3D(
            position=(0.0, -250.0, 100.0),
            look_target=(0.0, 0.0, 0.0),
            eye_up=(0.0, 0.0, 1.0),
        ),
        line_grid=rrb.LineGrid3D(
            visible=True,
            spacing=50.0,
            plane=rr.components.Plane3D.XY,
            color=[80, 80, 80, 100],
        ),
    )
    rr.send_blueprint(rrb.Blueprint(view))

    print(
        "\nRerun viewer launched.\n"
        "  Gold = skull reference geometry (perfectly stationary)\n"
        "  Blue/orange dots = eye centers (should overlap gold left_eye/right_eye keypoints)\n"
        "  Blue arrow  = left gaze in skull frame\n"
        "  Orange arrow = right gaze in skull frame\n"
        "Moving arrows = eye rotating relative to skull. Static = no eye-relative motion."
    )


if __name__ == "__main__":
    recording_folder = RecordingFolder.from_folder_path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording"
    )
    plot_gaze_in_skull_frame(recording_folder)
