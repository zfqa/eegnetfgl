$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$script = "run_bci2b_eegnetv4_confidence_window_subject1.py"
$saveDir = "results\bci2b_eegnetv4_confidence_window_1s_subject1"
$resultsCsv = "$saveDir\results.csv"

foreach ($subject in 2..9) {
    Write-Output "=== Starting subject $subject at $(Get-Date -Format o) ==="
    & $python $script `
        --subject $subject `
        --detector_win 1.0 `
        --detector_step 0.25 `
        --detect_start 0 `
        --detect_end 2 `
        --class_window 2 `
        --final_variant compact `
        --epochs_detector 300 `
        --epochs_classifier 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --seed 42 `
        --device cuda `
        --save_dir $saveDir `
        --results_csv $resultsCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Subject $subject failed with exit code $LASTEXITCODE."
    }
    Write-Output "=== Completed subject $subject at $(Get-Date -Format o) ==="
}
