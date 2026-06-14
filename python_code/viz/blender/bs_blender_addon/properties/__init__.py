"""Property groups for the BS Recorder addon.

Currently minimal — the ``recording_path`` property is attached directly
to ``bpy.types.Scene`` in ``__init__.py``.  This module exists for future
expansion (e.g. checkboxes to toggle which data layers to load).
"""

# PropertyGroup classes to register (empty for now — recording_path is
# a bare StringProperty on bpy.types.Scene).
BS_PROPERTY_GROUPS: list[type] = []
