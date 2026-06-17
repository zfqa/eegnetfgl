# Datasets for the Short-Window MI-EEG Experiments

This project uses public MOABB/BNCI motor-imagery datasets. The raw `.mat`
files are not committed to Git because the local cache is large and can be
re-created automatically by MOABB/MNE.

## Datasets

| Dataset in paper | MOABB dataset | BNCI code | Task | Subjects | Channels |
| --- | --- | --- | --- | --- | --- |
| BCI Competition IV-2a | `BNCI2014_001` | 001-2014 | 4-class MI | 1-9 | 22 EEG |
| BCI Competition IV-2b | `BNCI2014_004` | 004-2014 | 2-class MI | 1-9 | 3 EEG |

## Local cache used in this workspace

The current experiments used the following local MOABB/MNE cache:

```text
data/mne_data/MNE-bnci-data/~bci/database/001-2014/
data/mne_data/MNE-bnci-data/~bci/database/004-2014/
```

The cache contains the official training/evaluation `.mat` files downloaded by
MOABB/MNE. It is intentionally ignored by `.gitignore`:

```text
data/mne_data/
```

## Reproducing the data download

Install the Python dependencies and run any experiment script that loads
`BNCI2014_001` or `BNCI2014_004`. MOABB/MNE will download the files on first
use. To keep downloads inside the repository cache, set:

```powershell
$env:MNE_DATA = "data/mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data/mne_data"
```

Then run an experiment, for example:

```powershell
D:\anaconda\envs\eeg_fgl\python.exe run_fbcsp_lda_8_32_matrix.py --datasets bci2a bci2b --subjects 1 2 3 4 5 6 7 8 9 --seeds 42 2024 3407 7 123 --windows 2 4
```

## Versioning policy

Committed to Git:

- source code
- experiment runners
- manuscript drafts
- compact result summary CSV files needed for the paper tables

Not committed to Git:

- raw MOABB/MNE `.mat` downloads
- generated checkpoints
- generated diffusion checkpoints
- large intermediate outputs

