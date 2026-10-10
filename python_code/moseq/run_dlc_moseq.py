import keypoint_moseq as kpms
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = "/home/scholab/moseq/2d_dlc_behavior_test"
config = lambda: kpms.load_config(project_dir)

dlc_config = "/home/scholab/moseq/config.yaml"
kpms.setup_project(project_dir, deeplabcut_config=dlc_config)

video_directory = "/mnt/data/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording/mocap_data/synchronized_corrected_videos"

kpms.update_config(
    project_dir,
    video_dir=video_directory,
    anterior_bodyparts=["nose"],
    posterior_bodyparts=["tail_base"],
    use_bodyparts=["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_cam_tip", "right_cam_tip", "base"Remove , "spine_t1", "tail_base"],
    fps=90,
)

keypoint_data_path = "/mnt/data/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording/mocap_data/dlc_output/head_body_eyecam_retrain_test_v2/24676894_synchronized_correctedDLC_Resnet50_head_body_eyecam_retrain_test_v2_shuffle1_snapshot_best-90.csv"
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "deeplabcut")

kpms.update_config(project_dir, outlier_scale_factor=6.0)

coordinates, confidences = kpms.outlier_removal(
    coordinates,
    confidences,
    project_dir,
    overwrite=False,
    **config(),
)

data, metadata = kpms.format_data(coordinates, confidences, **config())

kpms.noise_calibration(project_dir, coordinates, confidences, **config())

plt.close("all")
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

kpms.update_config(
    project_dir,
    sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config()["fps"])
)

model = kpms.init_model(data, pca=pca, **config())
# model = kpms.update_hypparams(model, kappa=0.3)

num_ar_iters = 50
model, model_name = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    ar_only=True,
    num_iters=num_ar_iters,
)

model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir,
    model_name,
    iteration=num_ar_iters,
)

model = kpms.update_hypparams(model, kappa=1e4)

model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 500,
)

kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir,
    model_name,
)
results = kpms.extract_results(model, metadata, project_dir, model_name)
kpms.save_results_as_csv(results, project_dir, model_name)