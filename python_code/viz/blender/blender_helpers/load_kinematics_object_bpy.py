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


def load_rigid_body_kinematics_bpy(
    rbk: RigidBodyKinematics,
    translate: bool = True,
    scale: float = 1.0,
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
    mat: bpy.types.Material = get_or_create_material(
        name=f"{name}_rigid_body_material",
        base_color=(0.9, 0.6, 0.1, 1.0),  # warm gold / orange
        roughness=0.6,
    )
    for stick in stick_meshes:
        stick.data.materials.append(mat)
    print(f"  Created {len(stick_meshes)} edge stick meshes.")

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

    # ── Delegate animation to the existing RBK loader ──────────────────
    frame_empty: bpy.types.Object = load_rigid_body_kinematics_bpy(
        rbk=rbk,
        translate=False,  # eye rotates in place at origin
        scale=scale,
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
    _is_right: bool = "right" in rbk.name.lower()
    _base_color: tuple[float, float, float, float] = (
        (0.95, 0.15, 0.35, 1.0) if _is_right  # warm red-pink
        else (0.10, 0.55, 0.95, 1.0)  # cool blue
    )
    _mat: bpy.types.Material = get_or_create_material(
        name=f"{rbk.name}_eyeball_material",
        base_color=_base_color,
        roughness=0.4,
    )
    _sphere_mesh.materials.append(_mat)

    print(f"  Eyeball wireframe sphere '{_sphere_name}' ready.")
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
    )

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
    _cam_data.display_size = 0.002  # tiny viewport icon
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
    print(f"{'─' * 50}")
    return frame_empty
