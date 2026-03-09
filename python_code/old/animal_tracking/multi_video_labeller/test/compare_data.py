import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mimetypes

from pydantic import BaseModel, ConfigDict

from python_code.animal_tracking.multi_video_labeller.helpers.video_models import (
    ClickData,
    VideoMetadata,
    VideoPlaybackState,
)


class CompareData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    labelled_data: pd.DataFrame
    videos: list[VideoPlaybackState]

    @classmethod
    def from_paths(cls, labelled_data_csv: str | Path, video_folder: str | Path):
        dataframe = pd.read_csv(labelled_data_csv)
        dataframe = dataframe.set_index(["video", "frame"])
        print(f"loaded dataframe as: \n{dataframe.head()}")
        video_files = [
            f
            for f in os.listdir(video_folder)
            if mimetypes.guess_type(f)[0].startswith("video")
        ]
        video_paths = [str(Path(video_folder) / filename) for filename in video_files]

        if not video_files:
            raise ValueError(f"No videos found in {video_folder}")

        videos = []

        for video_name, video_path in zip(video_files, video_paths):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            metadata = VideoMetadata(
                path=video_path,
                name=video_name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )

            videos.append(
                VideoPlaybackState(
                    metadata=metadata,
                    cap=cap,
                )
            )

        return cls(labelled_data=dataframe, videos=videos)

    @staticmethod
    def draw_points(frame: np.ndarray, points: list[ClickData]):
        for point in points:
            cv2.circle(frame, (point.x, point.y), 5, (0, 0, 255), -1)

    @property
    def tracked_point_names(self) -> list[str]:
        tracked_point_names = set()
        for name in self.labelled_data.columns:
            name = name.strip("_x").strip("_y")
            tracked_point_names.add(name)
        return list(tracked_point_names)

    def get_click_data_by_video_frame(
        self, video_name: str, frame_number: int
    ) -> dict[str, ClickData]:
        print(f"video name: {video_name}, frame number: {frame_number}")
        video_index = [video.metadata.name for video in self.videos].index(video_name)
        series = self.labelled_data.loc[(video_name, frame_number)]
        click_data = {}
        for point_name in self.tracked_point_names:
            x = series[f"{point_name}_x"]
            y = series[f"{point_name}_y"]
            if not np.isnan(x) and not np.isnan(y):
                click_data[point_name] = ClickData(
                    video_index=video_index,
                    frame_number=frame_number,
                    video_x=x,
                    video_y=y,
                    window_x=x,
                    window_y=y,
                )

        return click_data

    def compare_frame(self, frame_number: int):
        for video in self.videos:
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.cap.read()
            if not ret:
                raise ValueError(
                    f"Could not read frame {frame_number} from video {video.metadata.name}"
                )
            points = self.get_click_data_by_video_frame(
                video.metadata.name, frame_number
            )
            self.draw_points(frame, list(points.values()))
            cv2.imshow(video.metadata.name, frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                continue

        cv2.destroyAllWindows()

    def close(self):
        for video in self.videos:
            video.cap.release()


if __name__ == "__main__":
    compare_data = CompareData.from_paths(
        labelled_data_csv="output.csv",
        video_folder=Path.home() / "freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos",
    )
    compare_data.compare_frame(0)
    compare_data.compare_frame(1)
    compare_data.compare_frame(2)
    compare_data.close()
