

import csv
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from python_code.eye_analysis.data_models.abase_model import FrozenABaseModel
from python_code.eye_analysis.data_models.trajectory_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


class CSVFormatError(ValueError):
    """Raised when CSV format is invalid or ambiguous."""
    pass


class CSVDataError(ValueError):
    """Raised when CSV data is malformed or missing required values."""
    pass


class ParsedTrajectoryData(FrozenABaseModel):
    """Strict container for parsed CSV data - no optionals allowed."""
    trajectories: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    frame_indices: np.ndarray
    timestamps: np.ndarray|None = None


class TrajectoryCSVLoader(BaseModel):
    """Loads 2D trajectory data from CSV files with ZERO tolerance for ambiguity.

    All parameters are REQUIRED. No defaults. No flexibility.
    NO TIMESTAMP INFERENCE - timestamps must be in CSV or provided explicitly.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoding: str
    min_confidence: float
    butterworth_cutoff: float
    butterworth_order: int
    timestamp_column_name: str

    def load_data(
            self,
            *,
            filepath: Path,
            name: str | None = None,
            timestamps: np.ndarray | list[float] | None = None,
    ) -> TrajectoryDataset:
        """Load CSV file into TrajectoryDataset.

        Args:
            filepath: Path to CSV file (must exist, must be .csv)
            timestamps: Optional timestamps array. Required if CSV has no timestamp column.

        Returns:
            TrajectoryDataset with validated data

        Raises:
            CSVFormatError: If format is invalid or ambiguous
            CSVDataError: If data is malformed or missing
            ValueError: If timestamps are missing from both CSV and parameter
        """
        if not filepath.exists():
            raise ValueError(f"File does not exist: {filepath}")

        if filepath.suffix.lower() != '.csv':
            raise ValueError(f"File must be .csv, got: {filepath.suffix}")

        format_type = self._detect_format(filepath=filepath)
        logger.info(f"Detected CSV format: {format_type}")

        if format_type == "dlc":
            parsed = self._read_dlc(filepath=filepath)
        elif format_type == "dlc_no_scorer":
            parsed = self._read_dlc_no_scorer(filepath=filepath)
        elif format_type == "tidy":
            parsed = self._read_tidy(filepath=filepath)
        elif format_type == "wide":
            parsed = self._read_wide(filepath=filepath)
        else:
            raise CSVFormatError(f"Unknown format: {format_type}")

        # Determine final timestamps: CSV or parameter
        final_timestamps: np.ndarray
        if parsed.timestamps is not None and timestamps is not None:
            # Both provided - fail to avoid ambiguity
            raise ValueError(
                f"Timestamps found in both CSV (column '{self.timestamp_column_name}') "
                f"and as parameter. Ambiguous - provide only one source."
            )
        elif parsed.timestamps is not None:
            final_timestamps = parsed.timestamps
            logger.info(f"Using timestamps from CSV column '{self.timestamp_column_name}'")
        elif timestamps is not None:
            final_timestamps = timestamps
            logger.info("Using timestamps from parameter")
        else:
            raise ValueError(
                f"No timestamps found. CSV has no '{self.timestamp_column_name}' column "
                f"and no timestamps parameter provided. Cannot proceed without timestamps."
            )

        # RIGID: timestamps MUST match frame count exactly
        if len(final_timestamps) != len(parsed.frame_indices):
            raise ValueError(
                f"Timestamp count ({len(final_timestamps)}) does not match "
                f"frame count ({len(parsed.frame_indices)})"
            )

        return TrajectoryDataset.create(
            dataset_name=name or filepath.stem,
            trajectories=parsed.trajectories,
            confidence=parsed.confidence,
            frame_indices=parsed.frame_indices,
            timestamps=final_timestamps,
            butterworth_cutoff=self.butterworth_cutoff,
            butterworth_order=self.butterworth_order,
        )

    def _detect_format(self, *, filepath: Path) -> str:
        """Detect CSV format - fails if ambiguous or unrecognized.

        Raises:
            CSVFormatError: If format cannot be determined unambiguously
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines: list[str] = []
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())

            f.seek(0)
            reader = csv.reader(f)
            header: list[str] = next(reader)

        if len(lines) < 2:
            raise CSVFormatError("CSV must have at least 2 rows")

        # Check for DLC format (3-row header)
        if len(lines) >= 3:
            row3: list[str] = lines[2].split(',')
            row3_lower = [coord.strip().lower() for coord in row3]
            if set(row3_lower) & {'x', 'y', 'likelihood'}:
                return "dlc"

        # Check for DLC without scorer (2-row header)
        if len(lines) >= 2:
            row2: list[str] = lines[1].split(',')
            row2_lower = [col.strip().lower() for col in row2]
            if set(row2_lower) & {'x', 'y', 'likelihood', 'confidence'}:
                return "dlc_no_scorer"

        # Check for tidy format
        header_lower: list[str] = [col.lower().strip() for col in header]
        required_tidy = {'frame', 'keypoint', 'x', 'y'}
        if required_tidy.issubset(set(header_lower)):
            return "tidy"

        # Check for wide format
        if any(col.strip().endswith('_x') for col in header):
            return "wide"

        raise CSVFormatError(
            f"Unrecognized CSV format. File: {filepath}\n"
            f"Supported formats: DLC (3-row), DLC no scorer (2-row), Tidy, Wide"
        )

    def _read_dlc(self, *, filepath: Path) -> ParsedTrajectoryData:
        """Read DeepLabCut format CSV with 3-row header.

        Looks for timestamp column in first column position.
        Fails loudly on any malformed data.
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise CSVFormatError("DLC format requires at least 4 rows (3 header + 1 data)")

        # Parse headers
        scorer_row = [col.strip() for col in lines[0].split(',')]
        bodypart_row = [col.strip() for col in lines[1].split(',')]
        coords_row = [col.strip() for col in lines[2].split(',')]

        if not (len(scorer_row) == len(bodypart_row) == len(coords_row)):
            logger.error(f"Error loading csv: {filepath} using DLC format")
            raise CSVFormatError("Header rows have mismatched lengths")

        # Check if first column is a label column (contains row headers)
        # DLC format typically has "scorer", "bodyparts", "coords" in first column
        is_label_column = (
                scorer_row[0].lower() in ['scorer', ''] and
                bodypart_row[0].lower() in ['bodyparts', 'bodypart'] and
                coords_row[0].lower() in ['coords', 'coord', 'coordinates']
        )

        # Check for timestamp column
        # If first column is labels, check second column for timestamps
        # Otherwise check first column
        has_timestamp_col = False
        timestamp_col_idx = -1

        if is_label_column:
            # Skip label column, check if second column is timestamp
            if len(bodypart_row) > 1 and bodypart_row[1].lower() == self.timestamp_column_name.lower():
                has_timestamp_col = True
                timestamp_col_idx = 1
                logger.info(f"Found timestamp column at index 1")
            start_idx = 1  # Skip label column
        else:
            # Check if first column is timestamp
            if bodypart_row[0].lower() == self.timestamp_column_name.lower():
                has_timestamp_col = True
                timestamp_col_idx = 0
                logger.info(f"Found timestamp column at index 0")
            start_idx = 0

        # If we found a timestamp column, skip it when building column map
        if has_timestamp_col:
            data_start_idx = timestamp_col_idx + 1
        else:
            data_start_idx = start_idx

        # Build column map (skip label and timestamp columns)
        column_map: dict[str, dict[str, int]] = {}

        for col_idx in range(data_start_idx, len(bodypart_row)):
            bodypart = bodypart_row[col_idx]
            coord = coords_row[col_idx]

            if bodypart:  # Non-empty bodypart name
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                column_map[bodypart][coord.lower()] = col_idx

        # Validate all bodyparts have x, y, likelihood
        valid_bodyparts: list[str] = []
        for bp, coords in column_map.items():
            if not {'x', 'y', 'likelihood'}.issubset(set(coords.keys())):
                logger.error(f"Error loading csv: {filepath} using DLC format")
                raise CSVFormatError(
                    f"Bodypart '{bp}' missing required columns. "
                    f"Has: {set(coords.keys())}, needs: {{x, y, likelihood}}"
                )
            valid_bodyparts.append(bp)

        if not valid_bodyparts:
            raise CSVFormatError("No valid bodyparts found in CSV")

        n_frames = len(lines) - 3
        trajectories: dict[str, np.ndarray] = {}
        confidence_data: dict[str, np.ndarray] = {}
        timestamps_array: np.ndarray | None = None

        if has_timestamp_col:
            timestamps_array = np.zeros(n_frames)

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            positions = np.zeros((n_frames, 2))
            confidence = np.zeros(n_frames)

            for i, line in enumerate(lines[3:]):
                values = line.split(',')

                # Extract timestamp if present
                if has_timestamp_col:
                    timestamps_array[i] = float(values[timestamp_col_idx].strip())

                # RIGID: No try-except, fail loudly
                x = float(values[coords['x']].strip())
                y = float(values[coords['y']].strip())
                conf = float(values[coords['likelihood']].strip())

                confidence[i] = conf


                positions[i] = [x, y]

            trajectories[bodypart] = positions
            confidence_data[bodypart] = confidence

        return ParsedTrajectoryData(
            trajectories=trajectories,
            confidence=confidence_data,
            frame_indices=np.arange(n_frames),
            timestamps=timestamps_array,
        )

    def _read_dlc_no_scorer(self, *, filepath: Path) -> ParsedTrajectoryData:
        """Read DLC format CSV with 2-row header.

        Looks for timestamp column in first column position.
        Fails loudly on any malformed data.
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise CSVFormatError("DLC no-scorer format requires at least 3 rows (2 header + 1 data)")

        bodypart_row = [col.strip() for col in lines[0].split(',')]
        coords_row = [col.strip() for col in lines[1].split(',')]

        if len(bodypart_row) != len(coords_row):
            raise CSVFormatError("Bodypart and coordinate rows have mismatched lengths")

        # Check if first column is a label column
        is_label_column = (
                bodypart_row[0].lower() in ['bodyparts', 'bodypart'] and
                coords_row[0].lower() in ['coords', 'coord', 'coordinates']
        )

        # Check for timestamp column
        has_timestamp_col = False
        timestamp_col_idx = -1

        if is_label_column:
            # Skip label column, check if second column is timestamp
            if len(bodypart_row) > 1 and bodypart_row[1].lower() == self.timestamp_column_name.lower():
                has_timestamp_col = True
                timestamp_col_idx = 1
                logger.info(f"Found timestamp column at index 1")
            start_idx = 1  # Skip label column
        else:
            # Check if first column is timestamp
            if bodypart_row[0].lower() == self.timestamp_column_name.lower():
                has_timestamp_col = True
                timestamp_col_idx = 0
                logger.info(f"Found timestamp column at index 0")
            start_idx = 0

        # If we found a timestamp column, skip it when building column map
        if has_timestamp_col:
            data_start_idx = timestamp_col_idx + 1
        else:
            data_start_idx = start_idx

        # Build column map (skip label and timestamp columns)
        column_map: dict[str, dict[str, int]] = {}

        for col_idx in range(data_start_idx, len(bodypart_row)):
            bodypart = bodypart_row[col_idx]
            coord = coords_row[col_idx]

            if bodypart:  # Non-empty bodypart name
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                coord_lower = coord.lower()
                if coord_lower in ['likelihood', 'confidence']:
                    column_map[bodypart]['likelihood'] = col_idx
                else:
                    column_map[bodypart][coord_lower] = col_idx

        # Validate all bodyparts have x, y, likelihood
        valid_bodyparts: list[str] = []
        for bp, coords in column_map.items():
            if not {'x', 'y', 'likelihood'}.issubset(set(coords.keys())):
                raise CSVFormatError(
                    f"Bodypart '{bp}' missing required columns. "
                    f"Has: {set(coords.keys())}, needs: {{x, y, likelihood}}"
                )
            valid_bodyparts.append(bp)

        if not valid_bodyparts:
            raise CSVFormatError("No valid bodyparts found in CSV")

        n_frames = len(lines) - 2
        trajectories: dict[str, np.ndarray] = {}
        confidence_data: dict[str, np.ndarray] = {}
        timestamps_array: np.ndarray | None = None

        if has_timestamp_col:
            timestamps_array = np.zeros(n_frames)

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            positions = np.zeros((n_frames, 2))
            confidence = np.zeros(n_frames)

            for i, line in enumerate(lines[2:]):
                values = line.split(',')

                # Extract timestamp if present
                if has_timestamp_col:
                    timestamps_array[i] = float(values[timestamp_col_idx].strip())

                # RIGID: No try-except, fail loudly
                x = float(values[coords['x']].strip())
                y = float(values[coords['y']].strip())
                conf = float(values[coords['likelihood']].strip())

                confidence[i] = conf

                # RIGID: Below threshold = invalid data
                if conf < self.min_confidence:
                    raise CSVDataError(
                        f"Frame {i}, bodypart '{bodypart}': confidence {conf:.3f} "
                        f"below minimum {self.min_confidence:.3f}"
                    )

                positions[i] = [x, y]

            trajectories[bodypart] = positions
            confidence_data[bodypart] = confidence

        return ParsedTrajectoryData(
            trajectories=trajectories,
            confidence=confidence_data,
            frame_indices=np.arange(n_frames),
            timestamps=timestamps_array,
        )
    def _read_tidy(self, *, filepath: Path) -> ParsedTrajectoryData:
        """Read tidy/long format CSV.

        Looks for timestamp column in the header.
        Fails if data is incomplete or inconsistent.
        """
        data_dict: dict[str, dict[int, np.ndarray]] = {}
        timestamps_dict: dict[int, float] = {}
        frame_set: set[int] = set()

        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            # Check if timestamp column exists
            has_timestamp_col = self.timestamp_column_name in reader.fieldnames

            for row_num, row in enumerate(reader, start=2):
                # RIGID: All fields must be present and parseable
                try:
                    frame = int(row['frame'])
                    keypoint = row['keypoint'].strip()
                    x = float(row['x'])
                    y = float(row['y'])
                except KeyError as e:
                    raise CSVDataError(f"Row {row_num}: Missing required column {e}")
                except ValueError as e:
                    raise CSVDataError(f"Row {row_num}: Cannot parse numeric value: {e}")

                if not keypoint:
                    raise CSVDataError(f"Row {row_num}: Empty keypoint name")

                frame_set.add(frame)

                # Extract timestamp if present
                if has_timestamp_col:
                    try:
                        timestamp = float(row[self.timestamp_column_name])
                        if frame in timestamps_dict:
                            # Verify consistency
                            if not np.isclose(timestamps_dict[frame], timestamp):
                                raise CSVDataError(
                                    f"Inconsistent timestamps for frame {frame}: "
                                    f"{timestamps_dict[frame]} vs {timestamp}"
                                )
                        else:
                            timestamps_dict[frame] = timestamp
                    except ValueError as e:
                        raise CSVDataError(
                            f"Row {row_num}: Cannot parse timestamp value: {e}"
                        )

                if keypoint not in data_dict:
                    data_dict[keypoint] = {}

                if frame in data_dict[keypoint]:
                    raise CSVDataError(
                        f"Duplicate entry for keypoint '{keypoint}' at frame {frame}"
                    )

                data_dict[keypoint][frame] = np.array([x, y])

        if not data_dict:
            raise CSVDataError("No trajectory data found in CSV")

        frame_indices = np.array(sorted(frame_set))
        n_frames = len(frame_indices)

        # Build timestamps array if present
        timestamps_array: np.ndarray | None = None
        if has_timestamp_col:
            if len(timestamps_dict) != n_frames:
                raise CSVDataError(
                    f"Timestamp count ({len(timestamps_dict)}) doesn't match "
                    f"frame count ({n_frames})"
                )
            timestamps_array = np.array([timestamps_dict[f] for f in frame_indices])
            logger.info(f"Found timestamps in column '{self.timestamp_column_name}'")

        # RIGID: All keypoints must have data for ALL frames
        trajectories: dict[str, np.ndarray] = {}
        for keypoint, frame_data in data_dict.items():
            if len(frame_data) != n_frames:
                raise CSVDataError(
                    f"Keypoint '{keypoint}' has {len(frame_data)} frames, "
                    f"expected {n_frames}"
                )

            positions = np.zeros((n_frames, 2))
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx not in frame_data:
                    raise CSVDataError(
                        f"Keypoint '{keypoint}' missing data for frame {frame_idx}"
                    )
                positions[i] = frame_data[frame_idx]

            trajectories[keypoint] = positions

        # Tidy format has no confidence data - use 1.0
        confidence_data = {name: np.ones(n_frames) for name in trajectories.keys()}

        return ParsedTrajectoryData(
            trajectories=trajectories,
            confidence=confidence_data,
            frame_indices=frame_indices,
            timestamps=timestamps_array,
        )

    def _read_wide(self, *, filepath: Path) -> ParsedTrajectoryData:
        """Read wide format CSV.

        Looks for timestamp column in the header.
        Fails if any keypoint is incomplete.
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            if not headers:
                raise CSVFormatError("CSV has no headers")

            # Check for timestamp column
            has_timestamp_col = self.timestamp_column_name in headers

            keypoint_names = {h[:-2] for h in headers if h.endswith('_x')}

            if not keypoint_names:
                raise CSVFormatError("No keypoints found (expected columns ending in '_x')")

            # RIGID: Check that all keypoints have both _x and _y
            for keypoint in keypoint_names:
                x_col = f"{keypoint}_x"
                y_col = f"{keypoint}_y"
                if y_col not in headers:
                    raise CSVFormatError(
                        f"Marker '{keypoint}' has '{x_col}' but missing '{y_col}'"
                    )

            rows = list(reader)

        if not rows:
            raise CSVDataError("CSV has no data rows")

        n_frames = len(rows)
        frame_indices = np.arange(n_frames)

        # Extract timestamps if present
        timestamps_array: np.ndarray | None = None
        if has_timestamp_col:
            timestamps_array = np.zeros(n_frames)
            for i, row in enumerate(rows):
                ts_str = row[self.timestamp_column_name].strip()
                if not ts_str:
                    raise CSVDataError(
                        f"Row {i + 2}: Empty timestamp value in column '{self.timestamp_column_name}'"
                    )
                timestamps_array[i] = float(ts_str)
            logger.info(f"Found timestamps in column '{self.timestamp_column_name}'")

        trajectories: dict[str, np.ndarray] = {}
        for keypoint in sorted(keypoint_names):
            x_col = f"{keypoint}_x"
            y_col = f"{keypoint}_y"

            positions = np.zeros((n_frames, 2))

            for i, row in enumerate(rows):
                # RIGID: No try-except, fail loudly
                x_str = row[x_col].strip()
                y_str = row[y_col].strip()

                if not x_str or not y_str:
                    raise CSVDataError(
                        f"Row {i + 2}, keypoint '{keypoint}': Empty value(s)"
                    )

                x = float(x_str)
                y = float(y_str)
                positions[i] = [x, y]

            trajectories[keypoint] = positions

        # Wide format has no confidence data - use 1.0
        confidence_data = {name: np.ones(n_frames) for name in trajectories.keys()}

        return ParsedTrajectoryData(
            trajectories=trajectories,
            confidence=confidence_data,
            frame_indices=frame_indices,
            timestamps=timestamps_array,
        )


def load_trajectory_dataset(
        *,
        filepath: Path,
        min_confidence: float,
        butterworth_cutoff: float,
        butterworth_order: int,
        timestamp_column_name: str = "timestamp",
        timestamps: np.ndarray | list[float] | None = None,
        encoding: str = "utf-8",
        name: str | None = None,
) -> TrajectoryDataset:

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

    if butterworth_order <= 0:
        raise ValueError(f"butterworth_order must be positive, got {butterworth_order}")

    if butterworth_cutoff <= 0:
        raise ValueError(f"butterworth_cutoff must be positive, got {butterworth_cutoff}")

    return TrajectoryCSVLoader(
        encoding=encoding,
        min_confidence=min_confidence,
        butterworth_cutoff=butterworth_cutoff,
        butterworth_order=butterworth_order,
        timestamp_column_name=timestamp_column_name,
    ).load_data(filepath=filepath, timestamps=timestamps, name=name)
