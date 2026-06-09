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
$script = "run_bci2b_detector_dual_teacher_subject1.py"
$saveDir = "results\bci2b_detector_dual_teacher_subject1"
$resultsCsv = Join-Path $saveDir "detector_dual_teacher_results.csv"

foreach ($subject in $Subjects) {
    if (Test-Path -LiteralPath $resultsCsv) {
        $existing = Import-Csv -LiteralPath $resultsCsv | Where-Object {
            [int]$_.subject -eq $subject -and
            [int]$_.seed -eq $Seed -and
            "$($_.strategy)" -eq "stable_peak_confidence" -and
            [int]$_.smooth_k -eq 3
        }
        if ($null -ne $existing) {
            Write-Output "Skip existing subject $subject."
            continue
        }
    }

    $detector = (
        "results\bci2b_eegnetv4_confidence_window_1s_subject1\" +
        "subject_${subject}\eegnetv4_05s_sliding_detector_best.pth"
    )
    $teacher = (
        "results\bci2b_eegnetv4_4s_teachers_seed42\" +
        "bci2b_subject_${subject}_eegnetv4_window_4_riemwin_4_seed_${Seed}_" +
        "eegkd_0_riekd_0_geo_0_supcon_0_spatial_0_filter_corr_sample_" +
        "geom_cosine_none_k_8_gt_1_gate_confidence_normalized_0_shrink_0.001_best.pth"
    )
    if (-not (Test-Path -LiteralPath $detector)) {
        throw "Detector not found: $detector"
    }
    if (-not (Test-Path -LiteralPath $teacher)) {
        throw "Teacher not found: $teacher"
    }

    Write-Output "`nRun subject $subject detector + dual teacher."
    & $python $script `
        --subject $subject `
        --seed $Seed `
        --detector_path $detector `
        --eegnet_teacher_path $teacher `
        --detector_win 1.0 `
        --detector_step 0.25 `
        --class_window 2.0 `
        --smooth_k 3 `
        --alpha_eegnet 0.4 `
        --alpha_riemann 0.4 `
        --temperature 1.0 `
        --epochs $Epochs `
        --patience $Patience `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --device cuda `
        --save_dir $saveDir
    if ($LASTEXITCODE -ne 0) {
        throw "Subject $subject failed with exit code $LASTEXITCODE."
    }
}

Write-Output "All requested subjects completed."
