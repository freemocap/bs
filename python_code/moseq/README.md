# Moseq

Run using the `keypoint_moseq` conda environment, not `bs`.

## Environment setup

Create the environment and install keypoint-moseq:

```bash
conda create -n keypoint_moseq python=3.10
conda activate keypoint_moseq
pip install keypoint-moseq[cuda]
```

Make the `bs` repo importable without installing its heavy dependencies:

```bash
echo "/home/scholab/Documents/repos/bs" > $(python -c "import site; print(site.getsitepackages()[0])")/bs_repo.pth
```