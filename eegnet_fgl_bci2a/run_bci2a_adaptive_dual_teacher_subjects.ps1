param(
    [int[]]$Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$Seed = 42,
    [int]$Epochs = 300,
    [int]$Patience = 100
)

$ErrorActionPreference = "Stop"
$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$script = "run_bci2a_adaptive_dual_teacher.py"
$saveDir = "results\bci2a_adaptive_dual_teacher_seed42"
$resultsCsv = Join-Path $saveDir "adaptive_dual_results.csv"

foreach ($subject in $Subjects) {
    if (Test-Path -LiteralPath $resultsCsv) {
        $existing = Import-Csv -LiteralPath $resultsCsv | Where-Object {
            [int]$_.subject -eq $subject -and [int]$_.seed -eq $Seed
        }
        if ($null -ne $existing) {
            Write-Output "Skip existing subject $subject."
            continue
        }
    }

    Write-Output "`nRun BCI2a subject $subject adaptive dual teacher."
    & $python $script `
        --subject $subject `
        --seed $Seed `
        --epochs $Epochs `
        --patience $Patience `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --alpha_eegnet 0.4 `
        --alpha_riemann 0.4 `
        --temperature 1.0 `
        --device cuda `
        --save_dir $saveDir
    if ($LASTEXITCODE -ne 0) {
        throw "Subject $subject failed with exit code $LASTEXITCODE."
    }
}

Write-Output "All requested BCI2a subjects completed."
