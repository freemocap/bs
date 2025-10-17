from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def create_visualizations(pupil_df:pd.DataFrame, analysis_dir:str):
    """Create and save all visualizations to HTML files"""
    pupil_outer_x = pupil_df["pupil_outer", "x"]
    pupil_outer_y = pupil_df["pupil_outer", "y"]
    pupil_outer_likelihood = pupil_df["pupil_outer", "likelihood"]
    presumed_framerate = 120
    presumed_timestamps = pd.Series(range(len(pupil_outer_x))) / presumed_framerate
    #Time Series Plot
    fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig_time.add_trace(go.Scattergl(x=presumed_timestamps,y=pupil_outer_x, mode="lines+markers", name="pupil_horizontal_position"), row=1, col=1)
    fig_time.add_trace(go.Scattergl(x=presumed_timestamps,y=pupil_outer_y, mode="lines+markers", name="pupil_vertical_position"), row=2, col=1)

    # Add axis labels
    fig_time.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig_time.update_yaxes(title_text="Position (pixels)", row=1, col=1)
    fig_time.update_yaxes(title_text="Position (pixels)", row=2, col=1)
    fig_time.update_layout(height=800, width=1280, title_text="Pupil Positions Time Series")
    time_series_path = Path(analysis_dir) / "gaze_trace.html"
    fig_time.write_html(str(time_series_path))

    fig_time.show()
    return [time_series_path]


def generate_gaze_trace_plots(dlc_pupil_csv: str):
    save_dir =  Path(dlc_pupil_csv).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    # Process data
    pupil_df = pd.read_csv(dlc_pupil_csv, header=[0,1])


    # Create visualizations
    html_files = create_visualizations(pupil_df,
                                       str(save_dir))
    print(f"Generated {len(html_files)} HTML files in {save_dir}")




if __name__ == "__main__":
    # Hardcoded default path
    DLC_PUPIL_CSV = r"./eye1DLC_Resnet50_dlc_pupil_tracking_shuffle1_snapshot_090.csv"


    generate_gaze_trace_plots(DLC_PUPIL_CSV)
