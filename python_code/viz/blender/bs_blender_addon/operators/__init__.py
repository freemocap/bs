"""Operators for the BS Recorder addon."""

from .clear_scene import BS_OT_clear_scene
from .load_recording import BS_OT_load_recording

BS_OPERATORS: list[type] = [
    BS_OT_clear_scene,
    BS_OT_load_recording,
]
