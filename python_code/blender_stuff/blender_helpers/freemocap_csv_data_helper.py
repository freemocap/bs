
import numpy as np
import pandas as pd





def filter_by_frame_df(df: pd.DataFrame, frame_number: int | list[int]) -> pd.DataFrame:
    """
    Filter the DataFrame by frame number.

    Args:
        df: DataFrame containing keypoint data
        frame_number: Single frame number or list of frame numbers to filter by

    Returns:
        Filtered DataFrame
    """
    if isinstance(frame_number, list):
        return df[df['frame'].isin(frame_number)]
    return df[df['frame'] == frame_number]


def filter_by_keypoint_df(df: pd.DataFrame, keypoint_name: str | list[str]) -> pd.DataFrame:
    """
    Filter the DataFrame by keypoint name.

    Args:
        df: DataFrame containing keypoint data
        keypoint_name: Single keypoint name or list of keypoint names to filter by

    Returns:
        Filtered DataFrame
    """
    if isinstance(keypoint_name, list):
        return df[df['keypoint'].isin(keypoint_name)]
    return df[df['keypoint'] == keypoint_name]


def filter_by_trajectory_df(df: pd.DataFrame, trajectory_type: str = 'rigid_3d_xyz') -> pd.DataFrame:
    """
    Filter the DataFrame by trajectory type.

    Args:
        df: DataFrame containing keypoint data
        trajectory_type: Trajectory type to filter by (default: 'rigid_3d_xyz')

    Returns:
        Filtered DataFrame
    """
    return df[df['trajectory'] == trajectory_type]


def extract_keypoint_data(
        df: pd.DataFrame,
        frame_number: int | list[int] | None = None,
        keypoint_name: str | list[str] | None = None,
        trajectory_type: str = 'rigid_3d_xyz'
) -> pd.DataFrame:
    """
    Extract keypoint data based on specified filters.

    Args:
        df: DataFrame containing keypoint data
        frame_number: Optional frame number(s) to filter by
        keypoint_name: Optional keypoint name(s) to filter by
        trajectory_type: Optional trajectory type to filter by (default: 'rigid_3d_xyz')

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    if frame_number is not None:
        filtered_df = filter_by_frame_df(filtered_df, frame_number)

    if keypoint_name is not None:
        filtered_df = filter_by_keypoint_df(filtered_df, keypoint_name)

    if trajectory_type is not None:
        filtered_df = filter_by_trajectory_df(filtered_df, trajectory_type)

    return filtered_df


def df_to_dict_of_arrays(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Convert a DataFrame of keypoint data to a nested dictionary of NumPy arrays.

    The resulting structure is:
    {
        keypoint_name: {
            'x': np.array([...]),
            'y': np.array([...]),
            'z': np.array([...]),
            'frame': np.array([...])
        },
        ...
    }

    Args:
        df: DataFrame containing keypoint data

    Returns:
        Dictionary of keypoint data as NumPy arrays
    """
    result = {}

    # Group by keypoint
    for keypoint, group in df.groupby('keypoint'):
        # Create a dictionary for this keypoint
        keypoint_data = {
            'x': group['x'].to_numpy(),
            'y': group['y'].to_numpy(),
            'z': group['z'].to_numpy(),
            'frame': group['frame'].to_numpy()
        }

        # Add additional columns if they exist
        if 'reprojection_error' in group.columns:
            keypoint_data['reprojection_error'] = group['reprojection_error'].to_numpy()

        result[keypoint] = keypoint_data

    return result


def filter_by_frame(
        data: dict[str, dict[str, np.ndarray]],
        frame_number: int | list[int]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Filter the dictionary of arrays by frame number.

    Args:
        data: Dictionary of keypoint data as NumPy arrays
        frame_number: Single frame number or list of frame numbers to filter by

    Returns:
        Filtered dictionary of keypoint data
    """
    result = {}

    for keypoint, keypoint_data in data.items():
        frames = keypoint_data['frame']

        # Create mask for the frames we want
        if isinstance(frame_number, list):
            mask = np.isin(frames, frame_number)
        else:
            mask = frames == frame_number

        # Only include this keypoint if it has data for the requested frames
        if np.any(mask):
            result[keypoint] = {
                key: values[mask] for key, values in keypoint_data.items()
            }

    return result


def filter_by_keypoint(
        data: dict[str, dict[str, np.ndarray]],
        keypoint_name: str | list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Filter the dictionary of arrays by keypoint name.

    Args:
        data: Dictionary of keypoint data as NumPy arrays
        keypoint_name: Single keypoint name or list of keypoint names to filter by

    Returns:
        Filtered dictionary of keypoint data
    """
    if isinstance(keypoint_name, list):
        return {k: v for k, v in data.items() if k in keypoint_name}
    return {keypoint_name: data[keypoint_name]} if keypoint_name in data else {}


def extract_keypoint_data_as_dict(
        df: pd.DataFrame,
        frame_number: int | list[int] | None = None,
        keypoint_name: str | list[str] | None = None,
        trajectory_type: str = 'rigid_3d_xyz'
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract keypoint data based on specified filters and return as a dictionary of NumPy arrays.

    Args:
        df: DataFrame containing keypoint data
        frame_number: Optional frame number(s) to filter by
        keypoint_name: Optional keypoint name(s) to filter by
        trajectory_type: Optional trajectory type to filter by (default: 'rigid_3d_xyz')

    Returns:
        Dictionary of keypoint data as NumPy arrays
    """
    # First filter using the DataFrame methods
    filtered_df = extract_keypoint_data(df, frame_number, keypoint_name, trajectory_type)

    # Convert to dictionary of arrays
    return df_to_dict_of_arrays(filtered_df)


if __name__ == "__main__":
    # Example usage
    filepath = "keypoints.csv"  # Update this with your actual file path
    df = load_keypoint_data(filepath)

    # Example 1: Get all rigid_3d_xyz data for frame 0
    frame0_rigid = extract_keypoint_data(df, frame_number=0)
    print("Example 1: All rigid_3d_xyz data for frame 0")
    print(frame0_rigid)
    print("\n" + "-" * 80 + "\n")

    # Example 2: Get nose keypoint data for all frames
    nose_data = extract_keypoint_data(df, keypoint_name="nose")
    print("Example 2: Nose keypoint data (rigid_3d_xyz) for all frames")
    print(nose_data)
    print("\n" + "-" * 80 + "\n")

    # Example 3: Get multiple keypoints for specific frames
    specific_data = extract_keypoint_data(
        df,
        frame_number=[0, 1],
        keypoint_name=["nose", "left_eye", "right_eye"]
    )
    print("Example 3: Specific keypoints for frames 0 and 1")
    print(specific_data)

    # Example 4: Using the dictionary-based functions
    print("\n" + "-" * 80 + "\n")
    print("Example 4: Dictionary-based extraction")
    dict_data = extract_keypoint_data_as_dict(
        df,
        frame_number=[0, 1],
        keypoint_name=["nose", "left_eye"]
    )
    print(f"Keys: {list(dict_data.keys())}")
    print(f"Data for nose: {dict_data['nose']}")