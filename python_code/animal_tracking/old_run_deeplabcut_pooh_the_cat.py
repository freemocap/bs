# %% setup
import deeplabcut

config_path = r"D:\deeplabcut-projects\pooh-the-cat-jsm-2024-03-01\config.yaml"


pooh_clips = [
    r"D:\deeplabcut-projects\pooh-the-cat-jsm-2024-02-28\videos\GH010071_clip16.mp4",    
]

# %% Extract frames

# deeplabcut.extract_frames(
#     config_path, mode="automatic", algo="uniform", userfeedback=False, crop=False
# )

# %% Label frames -

deeplabcut.label_frames(config_path)

# %% Check labels (print image with labels overlaid)

deeplabcut.check_labels(config_path, visualizeindividuals=True)

# %% Create training dataset

deeplabcut.create_training_dataset(
    config_path, net_type="resnet_50", augmenter_type="imgaug"
)

# THen go change the `[iteration-#]/train/pose_cfg.yml `max size` thing to a big number 1e6'll do it

# %% Train network

deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    max_snapshots_to_keep=5,
    autotune=True,
    displayiters=10,
    saveiters=1000,
    maxiters=300000,
    allow_growth=True,
)

# # %% Evaluate network
# deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)

# %% Analyze videos
deeplabcut.analyze_videos(config_path, pooh_clips, save_as_csv=True)

# %% Filter predictions

# deeplabcut.filterpredictions(
#     config_path,
#     pooh_clips,
#     shuffle=1,
#     trainingsetindex=0,
#     filtertype="arima",
#     p_bound=0.01,
#     ARdegree=3,
#     MAdegree=1,
#     alpha=0.01,
# )

# # %% Plot trajectories

# deeplabcut.plot_trajectories(config_path, pooh_clips)

# %% Create labeled video
deeplabcut.create_labeled_video(
    config_path,
    [pooh_clips[0]],
    trailpoints=1,
    pcutoff=0.1,
    draw_skeleton=True,
    videotype=".mp4",
    overwrite=True,
)

# %% Extract outlier frames
deeplabcut.extract_outlier_frames(
    config_path, pooh_clips, p_bound=0.0001, automatic=True
)


# %% Refine labels
deeplabcut.refine_labels(config_path)

# %% Merge datasets
deeplabcut.merge_datasets(config_path)


# %%
