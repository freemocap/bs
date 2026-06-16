# bs -  Blender Addon

Loads a recording folder into a Blender scene: keypoint stick figures, calibrated cameras with video planes, rigid-body kinematics, eye/gaze data.

### Install the BS Blender Addon to your local Blender

Run the `install_blender_addon.py` script in the `python_code/viz/blender/`  module (Note: it does not require a .venv, it uses only python built-ins)

```bash
python install_blender_addon.py
```

Creates a symlink into Blender's addons directory. On first run the addon auto-installs pip, Python dependencies, and git-sourced packages.

### Run in Blender

1. Open Blender 
2. Edit → Preferences → Add-ons → enable **"BS Recorder"**
3. 3D Viewport → sidebar (N) → **"BS"** tab
4. Pick a recording folder, click **Load Recording**

After clicking **Load Recording**, check the terminal for pipeline output and any error messages.

For Mac and Linux machines, you must launch Blender from a terminal to see the terminal logs. On Windows, you can select `Window >> Open Terminal Window` fro m the top bar menu after sartt up




### Run headless

```bash
./run_blender_viz.sh
```

Edit `RECORDING_PATH` in `python_code/viz/blender/__main_blender.py` first.

### Entry points

| File | Purpose |
|---|---|
| `python_code/viz/blender/__main_blender.py` | Headless entry point — path setup, deps, runs pipeline |
| `python_code/viz/blender/bs_blender_addon/__init__.py` | Blender addon — register/unregister, bootstrap deps |
| `python_code/viz/blender/blender_helpers/pipeline_runner.py` | `run_pipeline(path)` — shared between headless and addon |
| `install_blender_addon.py` | Cross-platform installer (Linux/macOS/Windows) |
| `run_blender_viz.sh` | Shell wrapper for headless Blender invocation |

### Key directories

```
python_code/
├── viz/blender/
│   ├── __main_blender.py          # Headless entry point
│   ├── bs_blender_addon/          # Blender addon (UI panel + operators)
│   │   ├── panels/                # Sidebar panel
│   │   ├── operators/             # Clear Scene, Load Recording
│   │   └── properties/            # Custom Blender properties
│   └── blender_helpers/           # Scene construction logic
│       ├── pipeline_runner.py     # Orchestrator
│       ├── blender_recording_model.py  # Pydantic data models
│       ├── create_blender_scene.py     # Scene assembly
│       ├── add_cameras.py         # Calibrated camera + video planes
│       ├── create_arena.py        # Wireframe bounding cube
│       └── load_simple_object/    # Stick-figure mesh construction
├── kinematics_core/               # Keypoint trajectories, rigid-body kinematics
├── batch_processing/              # Batch pipeline scripts
├── cameras/                       # Multicamera recording + postprocessing
└── utilities/                     # Video clipping, label tidying, etc.
```

### Dependencies

Managed automatically by `freemocap_blender_addon` on first run:

- **pip packages:** polars, pydantic, opencv-contrib-python, scipy, numpydantic, tabulate, toml, pyyaml
- **git sources:** bs, skellytracker, skellycam, freemocap (cloned to `~/.cache/freemocap/git_sources/`)

### Dev notes

- Active branch: `jon/dev`
- `__main_blender.py` purges `sys.modules` of edited `.py` files so Blender picks up changes without restarting
- `PYTHONDONTWRITEBYTECODE=1` avoids stale `.pyc` issues when editing outside Blender
