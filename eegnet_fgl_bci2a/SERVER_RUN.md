# Running the Current Main Experiments on a Linux Server

This note assumes a Linux server with Conda and an NVIDIA GPU.

## 1. Clone

```bash
git clone https://github.com/zfqa/eegnetfgl.git
cd eegnetfgl/eegnet_fgl_bci2a
```

## 2. Create environment

```bash
conda create -n eeg_fgl python=3.10 -y
conda activate eeg_fgl
pip install -r requirements.txt
```

If your server needs a specific CUDA PyTorch wheel, install PyTorch first from
the official PyTorch selector, then run `pip install -r requirements.txt`.

## 3. Set dataset cache

```bash
export MNE_DATA=data/mne_data
export MNE_DATASETS_BNCI_PATH=data/mne_data
export MNE_DONTWRITE_HOME=true
```

The raw BCI Competition IV-2a/2b files are downloaded by MOABB/MNE on first use.
They are not committed to Git.

## 4. Smoke test

Run one subject and one seed first:

```bash
SUBJECTS="1" SEEDS="42" EPOCHS=20 bash run_linux_bci2b_riemann_spatial.sh
```

Expected output file:

```text
results/bci2b_compact_riemann_spatial_2s4s_all9_linux/riemann_guided_results.csv
```

## 5. Full BCI IV-2b Riemann KD + spatial run

```bash
bash run_linux_bci2b_riemann_spatial.sh
```

This runs:

- dataset: BCI IV-2b / BNCI2014_004
- subjects: 1-9
- seeds: 42, 2024, 3407, 7, 123
- student: 2 s Compact EEGNet
- teacher: 4 s Riemann TS-LogReg
- preprocessing: 8-32 Hz, 0-4 s
- loss: CE + 0.1 Riemann KD + 0.2 spatial structure loss
- spatial source: pre-depthwise

## 6. Useful overrides

```bash
DEVICE=cpu SUBJECTS="1" SEEDS="42" EPOCHS=5 bash run_linux_bci2b_riemann_spatial.sh
PYTHON_BIN=/path/to/python SAVE_DIR=results/my_server_run bash run_linux_bci2b_riemann_spatial.sh
```

Logs are written to:

```text
results/linux_logs/bci2b_riemann_spatial/
```

