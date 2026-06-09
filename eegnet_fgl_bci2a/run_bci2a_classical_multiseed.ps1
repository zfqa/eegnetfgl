param(
    [int[]]$Seeds = @(42, 2024, 3407, 7, 123),
    [int[]]$Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9)
)

$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$root = "results\bci2a_classical_multiseed"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python @Arguments
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($pythonExitCode -ne 0) {
        throw "Command failed with exit code ${pythonExitCode}: $python $($Arguments -join ' ')"
    }
}

foreach ($seed in $Seeds) {
    $seedRoot = Join-Path $root "seed_$seed"
    $riemannDir = Join-Path $seedRoot "riemann"
    $cspDir = Join-Path $seedRoot "csp"
    foreach ($subject in $Subjects) {
        foreach ($window in @(2.0, 4.0)) {
            Invoke-CheckedPython @(
                "train_riemann_paper.py",
                "--dataset", "bci2a",
                "--subject", "$subject",
                "--window", "$window",
                "--seed", "$seed",
                "--save_dir", $riemannDir,
                "--methods", "mdm", "ts_logreg",
                "--estimators", "oas",
                "--alignments", "none",
                "--metric", "riemann",
                "--fmin", "8",
                "--fmax", "32",
                "--tmin", "0",
                "--tmax", "4"
            )
            Invoke-CheckedPython @(
                "train_csp_baseline.py",
                "--dataset", "bci2a",
                "--subject", "$subject",
                "--window", "$window",
                "--seed", "$seed",
                "--save_dir", $cspDir,
                "--method", "csp_lda",
                "--fmin", "8",
                "--fmax", "32",
                "--tmin", "0",
                "--tmax", "4"
            )
        }
    }
}

Write-Output "BCI2a classical five-seed matrix completed."
