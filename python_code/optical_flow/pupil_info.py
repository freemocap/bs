import json
from pathlib import Path
from typing import Tuple
from pydantic import BaseModel  

JSON_PATH = "code/optical_flow/pupil_info.json"

class PupilInfo(BaseModel):
    path: str
    recording_name: str
    eye_number: int
    crop: Tuple[int, int, int, int]

    @classmethod
    def from_path_and_crop(cls, path: Path | str, crop: Tuple[int, int, int, int]):
        path = Path(path)
        recording_name = path.parent.parent.stem
        eye_number = 0 if "eye0" in path.stem else 1
        return cls(path=str(path), recording_name=recording_name, eye_number=eye_number, crop=crop)
    
def save_models_to_json(models: dict[str, PupilInfo], path: Path):
    json_string = json.dumps({model_name: model.model_dump() for model_name, model in models.items()})
    path.write_text(json_string)
    
def load_json(path: Path) -> dict[str,PupilInfo]:
    loaded_dict = json.loads(path.read_text())
    return {model_name: PupilInfo(**model) for model_name, model in loaded_dict.items()}

def recording_id_from_path(path: Path) -> str:
    return path.parent.parent.stem + path.stem

def recording_id_from_model(model: PupilInfo) -> str:
    return model.recording_name + "eye" + str(model.eye_number)


if __name__ == "__main__":
    test_dict = {
        "model1": PupilInfo.from_path_and_crop(path="/path/to/model1/eye0.mp4",crop=(0, 0, 100, 100)),
        "model2": PupilInfo.from_path_and_crop(path="/path/to/model1/eye1.mp4", crop=(0, 0, 100, 100)),
    }
    save_models_to_json(test_dict, Path(JSON_PATH))
    loaded_dict = load_json(Path(JSON_PATH))
    print(loaded_dict)