import json
from pathlib import Path

with open(Path(__file__).parent / "video_rotations.json", 'r') as f:
    video_rotations = json.load(f)

print(video_rotations)