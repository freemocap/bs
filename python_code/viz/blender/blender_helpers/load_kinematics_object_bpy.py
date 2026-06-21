from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import bmesh
import bpy
import numpy as np

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.viz.blender.blender_helpers.load_simple_object.create_edge_stick_mesh import create_edge_stick_mesh
from python_code.viz.blender.blender_helpers.load_simple_object.get_or_create_material import get_or_create_material

if TYPE_CHECKING:
    from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics

MM_TO_M: float = 0.001
_BLENDER_VERSION: tuple[int, ...] = bpy.app.version
_IS_BLENDER_5: bool = _BLENDER_VERSION >= (5, 0, 0)
_IS_BLENDER_44_PLUS: bool = _BLENDER_VERSION >= (4, 4, 0)


def _get_or_create_fcurve(
    action: bpy.types.Action,
    anim_data: bpy.types.AnimData,
    data_path: str,
    index: int,
) -> bpy.types.FCurve:
    """Create an FCurve on *action*, handling Blender 4.x vs 5.x API differences.

    Blender 5.0+ uses a layered animation system:
        slot → layer → keyframe strip → channelbag → fcurve.

    Blender 4.4+ uses slotted actions where fcurves are still on the action
    but keyed to a slot via ``action_slot=``.

    Blender <4.4 uses bare ``action.fcurves.new(data_path, index)``.
    """
    if _IS_BLENDER_5:
        # --- layered animation (Blender 5.0+) ---
        if not action.slots:
            slot = action.slots.new(id_type='OBJECT', name=anim_data.id_data.name)
        else:
            slot = action.slots[0]

        if anim_data.action_slot is None:
            anim_data.action_slot = slot
        else:
            slot = anim_data.action_slot

        if not action.layers:
            layer = action.layers.new(name="Layer")
        else:
            layer = action.layers[0]

        if not layer.strips:
            strip = layer.strips.new(type='KEYFRAME')
        else:
            strip = layer.strips[0]

        channelbag = strip.channelbag(slot)
        if channelbag is None:
            channelbag = strip.channelbags.new(slot)

        return channelbag.fcurves.new(data_path=data_path, index=index)

    elif _IS_BLENDER_44_PLUS:
        # --- slotted actions (Blender 4.4+) ---
        if anim_data.action_slot is None:
            slot = action.slots.new(id_type="OBJECT", name=anim_data.id_data.name)
            anim_data.action_slot = slot

        return action.fcurves.new(
            data_path=data_path,
            index=index,
            action_slot=anim_data.action_slot,
        )

    else:
        # --- legacy (Blender <4.4) ---
        return action.fcurves.new(data_path=data_path, index=index)


def _bulk_keyframe_fcurve(
    fc: bpy.types.FCurve,
    frames: "np.ndarray",
    values: "np.ndarray",
) -> None:
    """Fill *fc* with keyframe points using the fast ``foreach_set`` API."""
    num_frames: int = len(frames)
    fc.keyframe_points.add(count=num_frames)

    co: np.ndarray = np.empty(num_frames * 2, dtype=np.float32)
    co[0::2] = frames.astype(np.float32)
    co[1::2] = values.astype(np.float32)
    fc.keyframe_points.foreach_set("co", co)
    fc.update()


def _keyframe_custom_properties(
    obj: bpy.types.Object,
    anim_data: bpy.types.AnimData,
    action: bpy.types.Action,
    frames: np.ndarray,
    properties: dict[str, tuple[np.ndarray, dict | None]],
) -> None:
    """Bulk-keyframe custom properties on *obj*.

    Parameters
    ----------
    properties:
        ``{name: (values, ui_meta)}`` where *values* is a ``(N,)``
        float64 array and *ui_meta* is an optional ``dict`` of kwargs
        for ``obj.id_properties_ui(name).update(**ui_meta)`` (min, max,
        description, etc.).
    """
    for prop_name, (values, ui_meta) in properties.items():
        # Initialise custom property
        obj[prop_name] = float(values[0])

        # Optional UI metadata (min / max / description)
        if ui_meta is not None:
            _ui = obj.id_properties_ui(prop_name)
            _ui.update(**ui_meta)

        # Create FCurve — scalar custom props use index=0
        fc: bpy.types.FCurve = _get_or_create_fcurve(
            action, anim_data, f'["{prop_name}"]', 0,
        )
        _bulk_keyframe_fcurve(fc, frames, values.astype(np.float32))


def _attach_rbk_custom_properties(
    frame_empty: bpy.types.Object,
    rbk: RigidBodyKinematics,
    anim_data: bpy.types.AnimData,
    action: bpy.types.Action,
    frames: np.ndarray,
) -> None:
    """Attach RBK scalar timeseries as custom properties on *frame_empty*."""
    _props: dict[str, tuple[np.ndarray, dict | None]] = {}

    # ── Position (mm) ────────────────────────────────────────────────
    _pos: np.ndarray = rbk.position_xyz
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"position_{_axis}"] = (_pos[:, _i], {"description": f"Position {_axis} (mm)"})

    # ── Velocity (mm/s) ──────────────────────────────────────────────
    _vel: np.ndarray = rbk.velocity_xyz
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"velocity_{_axis}"] = (_vel[:, _i], {"description": f"Velocity {_axis} (mm/s)"})

    # ── Acceleration (mm/s²) ─────────────────────────────────────────
    _acc: np.ndarray = rbk.acceleration_xyz
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"acceleration_{_axis}"] = (_acc[:, _i], {"description": f"Acceleration {_axis} (mm/s²)"})

    # ── Speed (mm/s) ─────────────────────────────────────────────────
    _props["speed"] = (rbk.speed.values, {"description": "Speed (mm/s)"})

    # ── Acceleration magnitude (mm/s²) ───────────────────────────────
    _props["acceleration_magnitude"] = (
        rbk.acceleration_magnitude.values,
        {"description": "Acceleration magnitude (mm/s²)"},
    )

    # ── Euler angles (rad) ───────────────────────────────────────────
    _euler: np.ndarray = rbk._euler_angles
    for _i, _name in enumerate(("roll", "pitch", "yaw")):
        _props[_name] = (_euler[:, _i], {"description": f"{_name.title()} (rad)", "soft_min": -6.28319, "soft_max": 6.28319})

    # ── Angular velocity global (rad/s) ──────────────────────────────
    _ang_vel: np.ndarray = rbk.angular_velocity_global
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"angular_velocity_{_axis}"] = (_ang_vel[:, _i], {"description": f"Angular velocity {_axis} (rad/s)"})

    # ── Angular speed (rad/s) ────────────────────────────────────────
    _props["angular_speed"] = (rbk.angular_speed.values, {"description": "Angular speed (rad/s)"})

    # ── Angular acceleration global (rad/s²) ─────────────────────────
    _ang_acc: np.ndarray = rbk.angular_acceleration_global
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"angular_acceleration_{_axis}"] = (_ang_acc[:, _i], {"description": f"Angular acceleration {_axis} (rad/s²)"})

    # ── Angular acceleration magnitude (rad/s²) ──────────────────────
    _props["angular_acceleration_magnitude"] = (
        rbk.angular_acceleration_magnitude.values,
        {"description": "Angular acceleration magnitude (rad/s²)"},
    )

    _keyframe_custom_properties(frame_empty, anim_data, action, frames, _props)
    _n: int = len(_props)
    print(f"  Custom properties: {_n} RBK timeseries attached to '{frame_empty.name}'")


def _attach_eye_custom_properties(
    frame_empty: bpy.types.Object,
    eye_kinematics: FerretEyeKinematics,
) -> None:
    """Attach FerretEyeKinematics-specific timeseries as custom properties."""
    _anim_data: bpy.types.AnimData = frame_empty.animation_data
    _action: bpy.types.Action = _anim_data.action
    _n_frames: int = eye_kinematics.n_frames
    _frames: np.ndarray = np.arange(_n_frames, dtype=np.float32)

    _props: dict[str, tuple[np.ndarray, dict | None]] = {}

    # ── Gaze angles (degrees) ─────────────────────────────────────────
    _props["gaze_azimuth_deg"] = (
        eye_kinematics.azimuth_degrees.astype(np.float64),
        {"description": "Gaze azimuth (deg, + = subject left)"},
    )
    _props["gaze_elevation_deg"] = (
        eye_kinematics.elevation_degrees.astype(np.float64),
        {"description": "Gaze elevation (deg, + = up)"},
    )

    # ── Anatomical angles (degrees) ───────────────────────────────────
    _props["adduction_deg"] = (
        np.degrees(eye_kinematics.adduction_angle.values.astype(np.float64)),
        {"description": "Adduction (deg, + = toward nose)"},
    )
    _props["elevation_deg"] = (
        np.degrees(eye_kinematics.elevation_angle.values.astype(np.float64)),
        {"description": "Elevation (deg, + = up)"},
    )
    _props["torsion_deg"] = (
        np.degrees(eye_kinematics.torsion_angle.values.astype(np.float64)),
        {"description": "Torsion (deg, + = extorsion)"},
    )

    # ── Anatomical velocities (deg/s) ─────────────────────────────────
    _props["adduction_velocity_deg_s"] = (
        np.degrees(eye_kinematics.adduction_velocity.values.astype(np.float64)),
        {"description": "Adduction velocity (deg/s)"},
    )
    _props["elevation_velocity_deg_s"] = (
        np.degrees(eye_kinematics.elevation_velocity.values.astype(np.float64)),
        {"description": "Elevation velocity (deg/s)"},
    )
    _props["torsion_velocity_deg_s"] = (
        np.degrees(eye_kinematics.torsion_velocity.values.astype(np.float64)),
        {"description": "Torsion velocity (deg/s)"},
    )

    # ── Anatomical accelerations (deg/s²) ─────────────────────────────
    _props["adduction_accel_deg_s2"] = (
        np.degrees(eye_kinematics.adduction_acceleration.values.astype(np.float64)),
        {"description": "Adduction acceleration (deg/s²)"},
    )
    _props["elevation_accel_deg_s2"] = (
        np.degrees(eye_kinematics.elevation_acceleration.values.astype(np.float64)),
        {"description": "Elevation acceleration (deg/s²)"},
    )
    _props["torsion_accel_deg_s2"] = (
        np.degrees(eye_kinematics.torsion_acceleration.values.astype(np.float64)),
        {"description": "Torsion acceleration (deg/s²)"},
    )

    # ── Socket landmarks (mm) ─────────────────────────────────────────
    _td: np.ndarray = eye_kinematics.socket_landmarks.tear_duct_mm
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"tear_duct_{_axis}"] = (_td[:, _i], {"description": f"Tear duct {_axis} (mm)"})

    _oe: np.ndarray = eye_kinematics.socket_landmarks.outer_eye_mm
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"outer_eye_{_axis}"] = (_oe[:, _i], {"description": f"Outer eye {_axis} (mm)"})

    _props["eye_opening_width_mm"] = (
        eye_kinematics.socket_landmarks.eye_opening_width_mm.values.astype(np.float64),
        {"description": "Eye opening width (mm)"},
    )

    # ── Tracked pupil center (mm) ─────────────────────────────────────
    _pc: np.ndarray = eye_kinematics.tracked_pupil.pupil_center_mm
    for _i, _axis in enumerate(("x", "y", "z")):
        _props[f"tracked_pupil_center_{_axis}"] = (_pc[:, _i], {"description": f"Tracked pupil center {_axis} (mm)"})

    # ── Pupil ellipse axes (mm) ───────────────────────────────────────
    _axes: np.ndarray = eye_kinematics.tracked_pupil.pupil_axes_mm
    _props["pupil_axis_major_mm"] = (_axes[:, 0], {"description": "Pupil major axis (mm)"})
    _props["pupil_axis_minor_mm"] = (_axes[:, 1], {"description": "Pupil minor axis (mm)"})

    _keyframe_custom_properties(frame_empty, _anim_data, _action, _frames, _props)
    _n: int = len(_props)
    print(f"  Custom properties: {_n} eye-specific timeseries attached to '{frame_empty.name}'")


def load_rigid_body_kinematics_bpy(
    rbk: RigidBodyKinematics,
    translate: bool = True,
    scale: float = 1.0,
    material_base_color: tuple[float, float, float, float] | None = None,
    cast_shadows: bool = True,
) -> bpy.types.Object:
    """Visualise *rbk* in the Blender scene using a transform hierarchy.

    Architecture
    ------------
    The ARROWS frame empty carries the rigid body's pose (position +
    orientation) across all frames.  Keypoint empties are **children** of
    that frame empty, placed at their **local** reference-geometry
    positions (no per-keypoint animation).  Blender's parent-child
    transform chain naturally computes ::

        world_keypoint = R(frame) @ local_keypoint + t(frame)

    which is exactly the rigid-body formula the solver uses.

    Edge stick meshes live at world level with Hook modifiers that
    track the keypoint empties' world positions.  They are NOT children
    of the frame empty — that would double-apply the transform (once
    via parent, once via Hook).

    Parameters
    ----------
    rbk: RigidBodyKinematics to visualise.
    translate: If ``False`` the frame empty only rotates (stays at
        world origin).
    scale: Uniform scale applied to keypoint local positions and the
        frame empty's display size.
    material_base_color: If provided, overrides the default gold/orange
        edge stick material color.  Useful for colour-coding by body
        part (e.g. red for right eye, blue for left eye).
    cast_shadows: If ``False``, sets ``visible_shadow = False`` on all
        created mesh objects so they don't occlude lights placed inside
        the reference frame.

    Returns
    -------
    frame_empty : bpy.types.Object
        The ARROWS empty that owns the body pose.
    """
    name: str = rbk.name
    n_frames: int = rbk.n_frames

    print(f"\n{'─' * 50}")
    print(f"Loading rigid body kinematics: {name}")
    print(f"  Frames: {n_frames}")
    print(f"  Keypoints: {rbk.keypoint_names}")
    print(f"  translate: {translate}, scale: {scale}")
    print(f"{'─' * 50}")

    # ── Safety check: detect MDS Z-mirror in reference geometry ───
    # Classical MDS has a 50 % chance of producing a Z-mirrored
    # geometry.  If a dorsal keypoint (one NOT defining the origin or
    # axes) has negative Z in a Z-up body frame, the reference
    # geometry is likely mirrored.  We flip Z as a band-aid so the
    # visualisation is anatomically correct, but the kinematics CSV
    # should be regenerated with the fixed ``estimate_reference_geometry``.
    _ref_geo = rbk.reference_geometry
    _origin_set: set[str] = set(_ref_geo.coordinate_frame.origin_keypoints)
    _axis_set: set[str] = set()
    for _ax in (
        _ref_geo.coordinate_frame.x_axis,
        _ref_geo.coordinate_frame.y_axis,
        _ref_geo.coordinate_frame.z_axis,
    ):
        if _ax is not None:
            _axis_set.update(_ax.keypoints)
    _frame_keypoints: set[str] = _origin_set | _axis_set

    _dorsal_name: str | None = None
    for _kp_name in rbk.keypoint_names:
        if _kp_name not in _frame_keypoints:
            _dorsal_name = _kp_name
            break

    _should_flip_z: bool = False
    if _dorsal_name is not None:
        _dorsal_z: float = float(_ref_geo.keypoints[_dorsal_name].z)
        if _dorsal_z < 0.0:
            _should_flip_z = True
            print()
            print("!" * 70)
            print("!  WARNING: Reference geometry Z-axis appears INVERTED")
            print("!")
            print(f"!  Keypoint '{_dorsal_name}' has Z = {_dorsal_z:.3f} mm")
            print("!  (negative Z in a Z-up body frame).")
            print("!")
            print("!  This is caused by MDS eigenvector sign ambiguity in")
            print("!  the reference geometry estimation.  The keypoint")
            print("!  local Z coordinates will be flipped so the")
            print("!  visualisation is anatomically correct.")
            print("!")
            print("!  ACTION: Re-run the solver pipeline with the updated")
            print("!  estimate_reference_geometry() to regenerate the")
            print("!  kinematics CSV and reference geometry JSON.")
            print("!" * 70)
            print()

    # ── Step 1: ARROWS frame empty (the body origin + orientation) ─
    frame_name: str = f"{name}_frame"
    print(f"  Creating ARROWS frame empty: '{frame_name}'")

    frame_empty: bpy.types.Object = bpy.data.objects.new(frame_name, None)
    frame_empty.empty_display_type = "ARROWS"
    frame_empty.empty_display_size = 0.1 * scale
    frame_empty.rotation_mode = "QUATERNION"
    bpy.context.collection.objects.link(frame_empty)

    # ── Step 2: Keypoint empties (children of frame_empty) ──────
    print(f"  Creating {len(rbk.reference_geometry.keypoints)} keypoint empties "
          f"(children of '{frame_name}')...")
    keypoint_empties: dict[str, bpy.types.Object] = {}

    for kp_name, marker_pos in rbk.reference_geometry.keypoints.items():
        local_pos_m: np.ndarray = marker_pos.to_array() * MM_TO_M * scale
        if _should_flip_z:
            local_pos_m[2] *= -1.0  # correct MDS Z-mirror

        kp_empty: bpy.types.Object = bpy.data.objects.new(
            f"{name}_{kp_name}_empty", None,
        )
        kp_empty.empty_display_type = "SPHERE"
        kp_empty.empty_display_size = 0.003 * scale
        kp_empty.location = tuple(local_pos_m)
        kp_empty.parent = frame_empty
        bpy.context.collection.objects.link(kp_empty)

        keypoint_empties[kp_name] = kp_empty

    _flip_note: str = " (Z-flipped)" if _should_flip_z else ""
    print(f"  Created {len(keypoint_empties)} keypoint empties "
          f"(static, local reference positions{_flip_note}).")

    # ── Origin marker sphere (child of frame_empty at [0,0,0]) ──
    _origin_radius: float = 0.005 * scale
    _origin_bm: bmesh.types.BMesh = bmesh.new()
    bmesh.ops.create_uvsphere(
        _origin_bm, u_segments=8, v_segments=6, radius=_origin_radius,
    )
    _origin_mesh: bpy.types.Mesh = bpy.data.meshes.new(
        f"{name}_origin_marker_mesh",
    )
    _origin_bm.to_mesh(_origin_mesh)
    _origin_bm.free()

    _origin_sphere: bpy.types.Object = bpy.data.objects.new(
        f"{name}_origin_marker", _origin_mesh,
    )
    _origin_sphere.parent = frame_empty
    _origin_sphere.location = (0.0, 0.0, 0.0)
    bpy.context.collection.objects.link(_origin_sphere)

    # Bright emission material for the origin marker
    _origin_mat: bpy.types.Material = bpy.data.materials.new(
        f"{name}_origin_marker_material",
    )
    _origin_mat.use_nodes = True
    _origin_nodes = _origin_mat.node_tree.nodes
    _origin_nodes.clear()
    _emit = _origin_nodes.new("ShaderNodeEmission")
    _emit.inputs["Color"].default_value = (1.0, 0.1, 0.1, 1.0)  # red
    _emit.inputs["Strength"].default_value = 3.0
    _out = _origin_nodes.new("ShaderNodeOutputMaterial")
    _origin_mat.node_tree.links.new(
        _emit.outputs["Emission"], _out.inputs["Surface"],
    )
    _origin_sphere.data.materials.append(_origin_mat)
    print(f"  Origin marker sphere (r={_origin_radius*1000:.1f}mm, red) "
          f"at body origin.")

    # ── Step 3: Edge stick meshes (world level, Hook → keypoints) ─
    edges: list[tuple[str, str]]
    if rbk.reference_geometry.display_edges:
        edges = list(rbk.reference_geometry.display_edges)
    else:
        edges = rbk.reference_geometry.get_rigid_edges()

    # Deduplicate (a,b) ↔ (b,a)
    deduped_edges: list[tuple[str, str]] = list({
        (min(a, b), max(a, b)) for a, b in edges
    })

    print(f"  Creating {len(deduped_edges)} edge stick meshes...")
    stick_meshes: list[bpy.types.Object] = []
    for name_a, name_b in deduped_edges:
        if name_a not in keypoint_empties or name_b not in keypoint_empties:
            print(f"    ⚠ Skipping edge ({name_a}, {name_b}) — keypoint not found")
            continue
        stick: bpy.types.Object = create_edge_stick_mesh(
            empty_a=keypoint_empties[name_a],
            empty_b=keypoint_empties[name_b],
            name=f"{name}_{name_a}_{name_b}_stick",
        )
        stick_meshes.append(stick)

    # Apply shared material
    _stick_color: tuple[float, float, float, float] = (
        material_base_color
        if material_base_color is not None
        else (0.9, 0.6, 0.1, 1.0)  # warm gold / orange (default)
    )
    mat: bpy.types.Material = get_or_create_material(
        name=f"{name}_rigid_body_material",
        base_color=_stick_color,
        roughness=0.6,
    )
    for stick in stick_meshes:
        stick.data.materials.append(mat)
    print(f"  Created {len(stick_meshes)} edge stick meshes.")

    # ── Shadow visibility ─────────────────────────────────────────
    if not cast_shadows:
        _origin_sphere.visible_shadow = False
        for stick in stick_meshes:
            stick.visible_shadow = False
        print(f"  Shadows disabled on origin marker + {len(stick_meshes)} stick meshes.")

    # ── Step 4: Keyframe frame_empty (position + rotation) ──────
    print(f"  Keyframing '{frame_name}' ({n_frames} frames)...")
    frames: np.ndarray = np.arange(n_frames, dtype=np.float32)

    anim_data: bpy.types.AnimData = frame_empty.animation_data_create()
    action: bpy.types.Action = bpy.data.actions.new(name=f"{frame_name}_action")
    anim_data.action = action

    # Location fcurves (if translate=True)
    if translate:
        positions_m: np.ndarray = rbk.position_xyz * MM_TO_M * scale
        for axis_idx in range(3):
            fc: bpy.types.FCurve = _get_or_create_fcurve(
                action, anim_data, "location", axis_idx,
            )
            _bulk_keyframe_fcurve(fc, frames, positions_m[:, axis_idx])
        print(f"    location fcurves: ✓")

    # Rotation quaternion fcurves (always)
    quaternions: np.ndarray = rbk.quaternions_wxyz
    for q_idx in range(4):
        fc = _get_or_create_fcurve(
            action, anim_data, "rotation_quaternion", q_idx,
        )
        _bulk_keyframe_fcurve(fc, frames, quaternions[:, q_idx])
    print(f"    rotation_quaternion fcurves: ✓")

    # ── Step 4.5: Attach RBK timeseries as custom properties ─────────
    _attach_rbk_custom_properties(frame_empty, rbk, anim_data, action, frames)

    # ── Step 5: Diagnostic — orientation & position at a mid-recording frame ─
    _diag_frame: int = n_frames // 2  # well past any frozen initial segment
    _q_diag = rbk.quaternions_wxyz[_diag_frame]
    _w, _x, _y, _z = (
        float(_q_diag[0]), float(_q_diag[1]),
        float(_q_diag[2]), float(_q_diag[3]),
    )

    _roll: float = math.atan2(
        2.0 * (_w * _x + _y * _z),
        1.0 - 2.0 * (_x * _x + _y * _y),
    )
    _pitch: float = math.asin(
        max(-1.0, min(1.0, 2.0 * (_w * _y - _z * _x))),
    )
    _yaw: float = math.atan2(
        2.0 * (_w * _z + _x * _y),
        1.0 - 2.0 * (_y * _y + _z * _z),
    )

    print(f"  Diagnostic (frame {_diag_frame} of {n_frames}):")
    print(f"    Euler angles: "
          f"roll={math.degrees(_roll):.2f}°, "
          f"pitch={math.degrees(_pitch):.2f}°, "
          f"yaw={math.degrees(_yaw):.2f}°")
    print(f"    Body origin (mm): "
          f"[{rbk.position_xyz[_diag_frame, 0]:.2f}, "
          f"{rbk.position_xyz[_diag_frame, 1]:.2f}, "
          f"{rbk.position_xyz[_diag_frame, 2]:.2f}]")

    # Eye midpoint computed from keypoint trajectories (should ≈ position_xyz)
    if "left_eye" in rbk.keypoint_names and "right_eye" in rbk.keypoint_names:
        _le = rbk.keypoint_trajectories["left_eye"][_diag_frame]
        _re = rbk.keypoint_trajectories["right_eye"][_diag_frame]
        _eye_mid = (_le + _re) / 2.0
        print(f"    Eye midpoint from trajectories (mm): "
              f"[{_eye_mid[0]:.2f}, {_eye_mid[1]:.2f}, {_eye_mid[2]:.2f}]")
        print(f"    (should equal body origin above)")

    # Base keypoint world position (for comparison with skull_and_spine)
    if "base" in rbk.keypoint_names:
        _base_world = rbk.keypoint_trajectories["base"][_diag_frame]
        _base_local = rbk.reference_geometry.keypoints["base"].to_array()
        _base_z_label: str = "Z"
        if _should_flip_z:
            _base_local[2] *= -1.0
            _base_z_label = "Z (flipped)"
        print(f"    Base keypoint local (mm): "
              f"[{_base_local[0]:.2f}, {_base_local[1]:.2f}, {_base_local[2]:.2f}]  "
              f"({_base_z_label})")
        print(f"    Base keypoint world (mm): "
              f"[{_base_world[0]:.2f}, {_base_world[1]:.2f}, {_base_world[2]:.2f}]")

    print(f"  ARROWS frame empty '{frame_name}' ready.")
    print(f"{'─' * 50}")
    return frame_empty


def load_eye_kinematics_bpy(
    eye_kinematics: FerretEyeKinematics,
    scale: float = 1.0,
) -> bpy.types.Object:
    """Visualise *eye_kinematics* in the Blender scene.

    Wraps :func:`load_rigid_body_kinematics_bpy` for the eyeball rigid
    body, then adds a wireframe UV-sphere mesh parented to the animated
    frame empty.  The eyeball reference geometry defines the north pole
    (+Z) as the rest gaze direction, so parenting the sphere to the
    rotation-animated empty naturally aligns the pole with the gaze.

    Parameters
    ----------
    eye_kinematics:
        ``FerretEyeKinematics`` whose ``.eyeball`` (a
        ``RigidBodyKinematics``) supplies the pose.
    scale:
        Uniform scale applied to the eyeball mesh radius.

    Returns
    -------
    frame_empty : bpy.types.Object
        The ARROWS empty that carries the eyeball pose.
    """
    rbk: RigidBodyKinematics = eye_kinematics.eyeball

    # ── Detect eye side for colour-coding ─────────────────────────────
    _is_right: bool = "right" in rbk.name.lower()
    _eye_color: tuple[float, float, float, float] = (
        (1.0, 0.0, 0.0, 1.0) if _is_right  # pure red
        else (0.0, 0.0, 1.0, 1.0)  # pure blue
    )

    # ── Delegate animation to the existing RBK loader ──────────────────
    frame_empty: bpy.types.Object = load_rigid_body_kinematics_bpy(
        rbk=rbk,
        translate=False,  # eye rotates in place at origin
        scale=scale,
        material_base_color=_eye_color,
        cast_shadows=False,
    )

    # ── Eye radius from reference geometry ─────────────────────────────
    _pupil_mm: np.ndarray = np.array(
        rbk.reference_geometry.keypoints["pupil_center"].to_array()
    )
    eye_radius_m: float = float(np.linalg.norm(_pupil_mm)) * MM_TO_M * scale

    print(f"  Eyeball wireframe sphere: radius = {eye_radius_m * 1000:.1f} mm")

    # ── Wireframe UV sphere mesh ───────────────────────────────────────
    _sphere_name: str = f"{rbk.name}_eyeball_wireframe"
    _sphere_mesh: bpy.types.Mesh = bpy.data.meshes.new(f"{_sphere_name}_mesh")
    _sphere_obj: bpy.types.Object = bpy.data.objects.new(_sphere_name, _sphere_mesh)

    _bm: bmesh.types.BMesh = bmesh.new()
    bmesh.ops.create_uvsphere(
        _bm, u_segments=8, v_segments=8, radius=eye_radius_m,
    )
    _bm.to_mesh(_sphere_mesh)
    _bm.free()

    # Wireframe modifier (same pattern as create_arena.py)
    _wire: bpy.types.WireframeModifier = _sphere_obj.modifiers.new(
        name="Wireframe", type="WIREFRAME",
    )
    _wire.thickness = 0.0002  # 0.2 mm on a ~3 mm sphere
    _wire.use_replace = True

    # Parent to frame_empty — inherits rotation, north pole tracks gaze
    _sphere_obj.parent = frame_empty
    _sphere_obj.location = (0.0, 0.0, 0.0)
    bpy.context.collection.objects.link(_sphere_obj)

    # ── Material ───────────────────────────────────────────────────────
    _mat: bpy.types.Material = get_or_create_material(
        name=f"{rbk.name}_eyeball_material",
        base_color=_eye_color,
        roughness=0.4,
    )
    _sphere_mesh.materials.append(_mat)

    # Disable shadow casting so spotlight at eye origin isn't occluded
    _sphere_obj.visible_shadow = False

    print(f"  Eyeball wireframe sphere '{_sphere_name}' ready.")

    # ── Attach eye-specific custom properties ─────────────────────────
    _attach_eye_custom_properties(frame_empty, eye_kinematics)

    print(f"{'─' * 50}")
    return frame_empty


def load_gaze_kinematics_bpy(
    gaze_kinematics: FerretEyeKinematics,
    skull_frame_empty: bpy.types.Object,
    eye_side: Literal["left", "right"] | None = None,
    horizontal_offset_deg: float = 0.0,
    vertical_offset_deg: float = 0.0,
    scale: float = 1.0,
) -> bpy.types.Object:
    """Visualise *gaze_kinematics* positioned at the skull eye with a camera.

    Gaze kinematics is in a **world reference frame** — the quaternion
    encodes the combined eye-in-head + head-in-world gaze direction.
    Unlike :func:`load_eye_kinematics_bpy` (eye-in-head only, at origin),
    this function:

    1. Animates the gaze rotation (same RBK delegate).
    2. Copies location from the skull's eye keypoint empty.
    3. Applies an optional angular offset (rest gaze calibration).
    4. Parents a miniature 7 mm camera that looks along the gaze vector.

    Hierarchy
    ---------
    ::

        frame_empty  (ARROWS, animated rotation, Copy Location → skull eye)
         ├── keypoint empties  (children)
         ├── edge sticks       (world-level, Hook → keypoints)
         ├── origin marker     (child at [0,0,0])
         └── offset_empty      (child, local rotation = angular offset)
              └── camera       (child, Ry(180°) so -Z = gaze +Z)

    Parameters
    ----------
    gaze_kinematics:
        ``FerretEyeKinematics`` whose ``.eyeball`` supplies the pose.
    skull_frame_empty:
        The ARROWS frame empty from ``load_rigid_body_kinematics_bpy``
        for the skull.  Its children include ``{skull_name}_{left,right}_eye_empty``
        keypoint empties.
    eye_side:
        Which eye this gaze belongs to.  Auto-detected from
        ``gaze_kinematics.name`` if *None*.
    horizontal_offset_deg:
        Angular offset (degrees) around +Y (gaze yaw / azimuth) applied
        **after** the gaze quaternion.  Positive → subject's left.
    vertical_offset_deg:
        Angular offset (degrees) around +X (gaze pitch / elevation)
        applied **after** the gaze quaternion.  Positive → up.
    scale:
        Uniform scale applied to the frame empty's display size.

    Returns
    -------
    frame_empty : bpy.types.Object
        The ARROWS empty that carries the gaze pose.
    """
    rbk: RigidBodyKinematics = gaze_kinematics.eyeball

    # ── Auto-detect eye side ──────────────────────────────────────────
    if eye_side is None:
        _name_lower: str = gaze_kinematics.name.lower()
        if "right" in _name_lower:
            eye_side = "right"
        elif "left" in _name_lower:
            eye_side = "left"
        else:
            raise ValueError(
                f"Cannot determine eye_side from name '{gaze_kinematics.name}'."
                f" Pass eye_side= explicitly."
            )

    # ── Determine eye colour ──────────────────────────────────────────
    _eye_color: tuple[float, float, float, float] = (
        (1.0, 0.0, 0.0, 1.0) if eye_side == "right"  # pure red
        else (0.0, 0.0, 1.0, 1.0)  # pure blue
    )

    print(f"\n{'─' * 50}")
    print(f"Loading gaze kinematics: {gaze_kinematics.name}")
    print(f"  eye_side: {eye_side}")
    print(f"  horizontal_offset: {horizontal_offset_deg}°")
    print(f"  vertical_offset: {vertical_offset_deg}°")
    print(f"{'─' * 50}")

    # ── Delegate animation to the existing RBK loader ──────────────────
    frame_empty: bpy.types.Object = load_rigid_body_kinematics_bpy(
        rbk=rbk,
        translate=False,  # positioned via Copy Location instead
        scale=scale,
        material_base_color=_eye_color,
        cast_shadows=False,
    )

    # ── Attach eye-specific custom properties ─────────────────────────
    _attach_eye_custom_properties(frame_empty, gaze_kinematics)

    # ── Copy Location from skull eye keypoint empty ────────────────────
    _skull_name: str = skull_frame_empty.name.removesuffix("_frame")
    _eye_empty_name: str = f"{_skull_name}_{eye_side}_eye_empty"
    _eye_empty: bpy.types.Object | None = None

    for _child in skull_frame_empty.children:
        if _child.name == _eye_empty_name:
            _eye_empty = _child
            break

    if _eye_empty is None:
        # Fallback: fuzzy match
        for _child in skull_frame_empty.children:
            if eye_side in _child.name.lower() and "eye" in _child.name.lower():
                _eye_empty = _child
                print(f"  ⚠ Exact name '{_eye_empty_name}' not found; "
                      f"using fuzzy match '{_eye_empty.name}'")
                break

    if _eye_empty is None:
        _available: list[str] = [c.name for c in skull_frame_empty.children]
        raise ValueError(
            f"Cannot find skull eye keypoint empty. "
            f"Tried exact '{_eye_empty_name}' and fuzzy '{eye_side}.*eye'. "
            f"Available children: {_available}"
        )

    _cl: bpy.types.CopyLocationConstraint = frame_empty.constraints.new(
        "COPY_LOCATION"
    )
    _cl.target = _eye_empty
    print(f"  Copy Location: '{frame_empty.name}' → '{_eye_empty.name}'")

    # ── Angular offset empty ───────────────────────────────────────────
    _h_rad: float = math.radians(horizontal_offset_deg)
    _v_rad: float = math.radians(vertical_offset_deg)

    # Q_offset = Q_y(horizontal) * Q_x(vertical)
    # Q_y(θ) = [cos(θ/2), 0, sin(θ/2), 0]  (w,x,y,z)
    # Q_x(φ) = [cos(φ/2), sin(φ/2), 0, 0]
    _q_y: tuple[float, float, float, float] = (
        math.cos(_h_rad / 2), 0.0, math.sin(_h_rad / 2), 0.0,
    )
    _q_x: tuple[float, float, float, float] = (
        math.cos(_v_rad / 2), math.sin(_v_rad / 2), 0.0, 0.0,
    )
    # Multiply: q_offset = q_y * q_x
    _qw: float = (_q_y[0] * _q_x[0] - _q_y[1] * _q_x[1]
                  - _q_y[2] * _q_x[2] - _q_y[3] * _q_x[3])
    _qx: float = (_q_y[0] * _q_x[1] + _q_y[1] * _q_x[0]
                  + _q_y[2] * _q_x[3] - _q_y[3] * _q_x[2])
    _qy: float = (_q_y[0] * _q_x[2] - _q_y[1] * _q_x[3]
                  + _q_y[2] * _q_x[0] + _q_y[3] * _q_x[1])
    _qz: float = (_q_y[0] * _q_x[3] + _q_y[1] * _q_x[2]
                  - _q_y[2] * _q_x[1] + _q_y[3] * _q_x[0])

    _offset_name: str = f"{rbk.name}_gaze_offset"
    _offset_empty: bpy.types.Object = bpy.data.objects.new(_offset_name, None)
    _offset_empty.empty_display_type = "PLAIN_AXES"
    _offset_empty.empty_display_size = 0.005 * scale
    _offset_empty.rotation_mode = "QUATERNION"
    _offset_empty.rotation_quaternion = (_qw, _qx, _qy, _qz)
    _offset_empty.parent = frame_empty
    bpy.context.collection.objects.link(_offset_empty)

    if horizontal_offset_deg != 0.0 or vertical_offset_deg != 0.0:
        print(f"  Angular offset empty '{_offset_name}': "
              f"Q_y({horizontal_offset_deg}°) * Q_x({vertical_offset_deg}°)")

    # ── Miniature camera (child of offset empty) ───────────────────────
    _cam_name: str = f"{rbk.name}_gaze_camera"
    _cam_data: bpy.types.Camera = bpy.data.cameras.new(_cam_name)
    _cam_data.lens = 7.0
    _cam_data.lens_unit = "MILLIMETERS"
    _cam_data.display_size = 0.008  # visible in viewport
    _cam_data.sensor_fit = "HORIZONTAL"

    _cam_obj: bpy.types.Object = bpy.data.objects.new(_cam_name, _cam_data)
    _cam_obj.rotation_mode = "QUATERNION"
    # Ry(180°) — flips camera -Z to gaze +Z while keeping Y aligned
    _cam_obj.rotation_quaternion = (0.0, 0.0, 1.0, 0.0)  # w,x,y,z
    _cam_obj.parent = _offset_empty
    bpy.context.collection.objects.link(_cam_obj)

    print(f"  Camera '{_cam_name}': 7 mm lens, parented to '{_offset_name}'")
    print(f"  Camera rotation: Ry(180°) so -Z = gaze +Z, +Y = up")
    print(f"  Hierarchy: {frame_empty.name} → {_offset_name} → {_cam_name}")

    # ── Spotlight (coloured, wide cone, follows gaze) ──────────────────
    _light_name: str = f"{rbk.name}_gaze_spotlight"
    _light_data: bpy.types.Light = bpy.data.lights.new(name=_light_name, type='SPOT')
    _light_data.spot_size = math.pi  # 180° cone (Blender max, full angle edge-to-edge)
    _light_data.color = (1.0, 0.0, 0.0) if eye_side == "right" else (0.0, 0.0, 1.0)
    _light_data.energy = 50.0

    _light_obj: bpy.types.Object = bpy.data.objects.new(_light_name, _light_data)
    _light_obj.rotation_mode = "QUATERNION"
    # Ry(180°) — flips default -Z spotlight to shine along +Z (gaze direction)
    _light_obj.rotation_quaternion = (0.0, 0.0, 1.0, 0.0)  # w,x,y,z
    _light_obj.parent = _offset_empty  # follows gaze + angular offset
    bpy.context.collection.objects.link(_light_obj)

    print(f"  Spotlight '{_light_name}': {eye_side} eye, 180° cone, "
          f"parented to '{_offset_name}'")
    print(f"{'─' * 50}")
    return frame_empty
