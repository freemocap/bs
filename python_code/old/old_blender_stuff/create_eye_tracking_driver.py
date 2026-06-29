"""Process eye tracking data and create Blender drivers."""

from pathlib import Path
import numpy as np
import bpy

from python_code.blender_stuff.load_trajectories import load_trajectories_from_dlc_csv


def calculate_pupil_center(
        *,
        filepath: Path | str,
        pupil_point_names: list[str] | None = None,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
) -> np.ndarray:
    """
    Calculate pupil center from eye tracking points.
    
    Args:
        filepath: Path to DLC CSV with eye tracking data
        pupil_point_names: List of point names to average (default: ['p1'-'p8'])
        scale_factor: Scale multiplier for coordinates
        z_value: Z-coordinate for 2D data
        
    Returns:
        Array of shape (n_frames, 3) with pupil center coordinates
    """
    if pupil_point_names is None:
        pupil_point_names = [f'p{i}' for i in range(1, 9)]
    
    print(f"Loading eye tracking data from: {Path(filepath).name}")
    
    # Load all eye tracking points
    trajectories = load_trajectories_from_dlc_csv(
        filepath=filepath,
        scale_factor=scale_factor,
        z_value=z_value,
    )
    
    # Extract pupil points and calculate mean
    pupil_points = []
    for point_name in pupil_point_names:
        if point_name in trajectories:
            pupil_points.append(trajectories[point_name])
        else:
            print(f"Warning: '{point_name}' not found in tracking data")
    
    if not pupil_points:
        raise ValueError(f"No pupil points found. Available points: {list(trajectories.keys())}")
    
    # Stack and compute mean across points (axis 0)
    pupil_stack = np.stack(pupil_points, axis=0)  # (n_points, n_frames, 3)
    pupil_center = np.nanmean(pupil_stack, axis=0)  # (n_frames, 3)
    
    print(f"✓ Calculated pupil center from {len(pupil_points)} points")
    
    return pupil_center


def calculate_horopter_movement(
        *,
        pupil_center: np.ndarray,
        axis: int = 0,
) -> np.ndarray:
    """
    Convert 2D/3D pupil center to 1D movement along horopter (typically horizontal).
    
    Args:
        pupil_center: Array of shape (n_frames, 3) with pupil positions
        axis: Which axis represents horizontal movement (0=x, 1=y, 2=z)
        
    Returns:
        Array of shape (n_frames,) with 1D displacement from mean position
    """
    # Extract movement along specified axis
    axis_movement = pupil_center[:, axis]
    
    # Calculate displacement from mean position
    mean_position = np.nanmean(axis_movement)
    displacement = axis_movement - mean_position
    
    print(f"✓ Calculated horopter displacement (axis={axis})")
    print(f"  Mean position: {mean_position:.4f}")
    print(f"  Range: [{np.nanmin(displacement):.4f}, {np.nanmax(displacement):.4f}]")
    
    return displacement


def create_object_with_custom_property(
        *,
        name: str,
        property_name: str = "horopter_value",
        parent_object: bpy.types.Object | None = None,
        empty_type: str = "SPHERE",
        empty_scale: float = 0.01,
) -> bpy.types.Object:
    """
    Create an empty object with a custom property for storing animated values.
    
    Args:
        name: Name for the object
        property_name: Name of the custom property
        parent_object: Optional parent object
        empty_type: Blender empty display type
        empty_scale: Display size of empty
        
    Returns:
        Created Blender object
    """
    # Create empty object
    obj = bpy.data.objects.new(name=name, object_data=None)
    obj.empty_display_type = empty_type
    obj.empty_display_size = empty_scale
    
    if parent_object is not None:
        obj.parent = parent_object
    
    bpy.context.collection.objects.link(object=obj)
    
    # Add custom property
    obj[property_name] = 0.0
    
    # Make it animatable and set property settings
    id_props = obj.id_properties_ui(property_name)
    id_props.update(
        description=f"Animated {property_name} data",
        default=0.0,
        min=-1000.0,
        max=1000.0,
        soft_min=-10.0,
        soft_max=10.0,
    )
    
    print(f"✓ Created object '{name}' with custom property '{property_name}'")
    
    return obj


def keyframe_custom_property(
        *,
        obj: bpy.types.Object,
        property_name: str,
        values: np.ndarray,
        start_frame: int = 0,
) -> None:
    """
    Keyframe a custom property with time-series data.
    
    Args:
        obj: Blender object with the custom property
        property_name: Name of the property to keyframe
        values: Array of shape (n_frames,) with values to keyframe
        start_frame: First frame number
    """
    if property_name not in obj:
        raise ValueError(f"Object '{obj.name}' does not have property '{property_name}'")
    
    num_frames = len(values)
    
    # Create action for animation
    if obj.animation_data is None:
        obj.animation_data_create()
    
    if obj.animation_data.action is None:
        action = bpy.data.actions.new(name=f"{obj.name}_Action")
        obj.animation_data.action = action
    else:
        action = obj.animation_data.action
    
    # Handle Blender 4.4+ action structure
    if bpy.app.version >= (4, 4):
        # Find or create slot
        if len(action.slots) == 0:
            slot = action.slots.new(id_type='OBJECT', name=obj.name)
        else:
            slot = action.slots[0]
        
        # Find or create layer and strip
        if len(action.layers) == 0:
            layer = action.layers.new(name="Layer")
            strip = layer.strips.new(type='KEYFRAME')
        else:
            layer = action.layers[0]
            strip = layer.strips[0] if len(layer.strips) > 0 else layer.strips.new(type='KEYFRAME')
        
        channelbag = strip.channelbag(slot=slot, ensure=True)
        obj.animation_data.action_slot = slot
        
        # Create fcurve
        data_path = f'["{property_name}"]'
        fcurve = channelbag.fcurves.new(data_path=data_path)
    else:
        # Blender < 4.4
        data_path = f'["{property_name}"]'
        fcurve = action.fcurves.new(data_path=data_path)
    
    # Add keyframes
    fcurve.keyframe_points.add(count=num_frames)
    
    # Set keyframe data efficiently
    frames = np.arange(start_frame, start_frame + num_frames, dtype=np.float32)
    co = np.empty(shape=(2 * num_frames,), dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values.astype(np.float32)
    
    fcurve.keyframe_points.foreach_set("co", co)
    fcurve.update()
    
    print(f"✓ Keyframed '{property_name}' on '{obj.name}' ({num_frames} frames)")


def create_driver_on_constraint(
        *,
        target_object: bpy.types.Object,
        constraint_name: str,
        constraint_property: str,
        driver_object: bpy.types.Object,
        driver_property: str,
        expression: str = "var",
        multiplier: float = 1.0,
) -> bpy.types.FCurve:
    """
    Create a driver that controls a constraint property based on a custom property.
    
    Args:
        target_object: Object with the constraint
        constraint_name: Name of the constraint
        constraint_property: Property on constraint to drive (e.g., "influence")
        driver_object: Object with the custom property
        driver_property: Name of the custom property to read
        expression: Driver expression (default: "var", can be "var * 2", etc.)
        multiplier: Simple multiplier for the value (alternative to expression)
        
    Returns:
        Created driver FCurve
    """
    # Get constraint
    if constraint_name not in target_object.constraints:
        raise ValueError(f"Constraint '{constraint_name}' not found on '{target_object.name}'")
    
    constraint = target_object.constraints[constraint_name]
    
    # Create driver
    data_path = f'constraints["{constraint_name}"].{constraint_property}'
    fcurve = target_object.driver_add(data_path=data_path)
    
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    
    # Add variable
    var = driver.variables.new()
    var.name = "var"
    var.type = 'SINGLE_PROP'
    
    # Set variable target
    var.targets[0].id = driver_object
    var.targets[0].data_path = f'["{driver_property}"]'
    
    # Set expression
    if multiplier != 1.0:
        driver.expression = f"var * {multiplier}"
    else:
        driver.expression = expression
    
    print(f"✓ Created driver: {target_object.name}.{constraint_name}.{constraint_property} ← {driver_object.name}['{driver_property}']")
    
    return fcurve


def create_eye_tracking_controller(
        *,
        eye_csv_path: Path | str,
        parent_object: bpy.types.Object,
        name: str = "pupil_center",
        scale_factor: float = 0.001,
        horopter_axis: int = 0,
        start_frame: int = 0,
) -> tuple[bpy.types.Object, np.ndarray]:
    """
    Complete workflow: load eye data, calculate horopter movement, create controller object.
    
    Args:
        eye_csv_path: Path to DLC eye tracking CSV
        parent_object: Parent object for organization
        name: Name for controller object
        scale_factor: Scale multiplier for coordinates
        horopter_axis: Axis for horizontal movement (0=x, 1=y)
        start_frame: First frame number
        
    Returns:
        Tuple of (controller object, horopter displacement array)
    """
    print(f"\n{'='*60}")
    print(f"Creating eye tracking controller: {name}")
    print(f"{'='*60}\n")
    
    # 1. Calculate pupil center
    pupil_center = calculate_pupil_center(
        filepath=eye_csv_path,
        scale_factor=scale_factor,
        z_value=0.0,
    )
    
    # 2. Calculate horopter movement
    horopter_displacement = calculate_horopter_movement(
        pupil_center=pupil_center,
        axis=horopter_axis,
    )
    
    # 3. Create controller object with custom property
    controller = create_object_with_custom_property(
        name=name,
        property_name="horopter_displacement",
        parent_object=parent_object,
        empty_type="PLAIN_AXES",
        empty_scale=0.02,
    )
    
    # 4. Keyframe the custom property
    keyframe_custom_property(
        obj=controller,
        property_name="horopter_displacement",
        values=horopter_displacement,
        start_frame=start_frame,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Eye tracking controller created!")
    print(f"{'='*60}\n")
    
    return controller, horopter_displacement



def create_eye_movement_visualization(
        *,
        controller: bpy.types.Object,
        parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Create a simple object that visualizes eye movement using a driver.
    
    This is an example - modify to suit your needs!
    
    Args:
        controller: Object with horopter_displacement property
        parent: Parent object for organization
        
    Returns:
        Created visualization object
    """
    
    
    # Create a sphere to represent the pupil/gaze point
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, location=(0, 0, 0))
    viz_obj = bpy.context.active_object
    viz_obj.name = "pupil_visualization"
    viz_obj.parent = parent
    
    # Add a Limit Location constraint
    constraint = viz_obj.constraints.new(type='LIMIT_LOCATION')
    constraint.name = "horopter_limit"
    constraint.use_min_x = True
    constraint.use_max_x = True
    constraint.owner_space = 'LOCAL'
    
    # Drive the constraint's min_x based on horopter displacement
    # This will move the sphere left/right based on eye movement
    fcurve = viz_obj.driver_add(f'constraints["{constraint.name}"].min_x')
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    
    var = driver.variables.new()
    var.name = "horopter"
    var.type = 'SINGLE_PROP'
    var.targets[0].id = controller
    var.targets[0].data_path = '["horopter_displacement"]'
    
    # Scale the movement (adjust multiplier as needed)
    driver.expression = "horopter * 0.1"
    
    # Also drive max_x to same value
    fcurve2 = viz_obj.driver_add(f'constraints["{constraint.name}"].max_x')
    driver2 = fcurve2.driver
    driver2.type = 'SCRIPTED'
    
    var2 = driver2.variables.new()
    var2.name = "horopter"
    var2.type = 'SINGLE_PROP'
    var2.targets[0].id = controller
    var2.targets[0].data_path = '["horopter_displacement"]'
    
    driver2.expression = "horopter * 0.1"
    
    print(f"✓ Created eye movement visualization with driver")
    
    return viz_obj
