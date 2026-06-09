param(
    [int[]]$Seeds = @(42, 2024, 3407, 7, 123),
    [int[]]$Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$Epochs = 300,
    [int]$Patience = 100
)

$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$script = "train_student_riemann_guided.py"
$root = "results\bci2a_eegnetv4_strong_multiseed"

function Test-FloatEqual {
    param($Actual, [double]$Expected)
    if ($null -eq $Actual -or "$Actual" -eq "") {
        return $false
    }
    return [math]::Abs([double]$Actual - $Expected) -lt 1e-8
}

function Test-GuidedResult {
    param(
        [string[]]$Directories,
        [int]$Subject,
        [int]$Seed,
        [double]$Window,
        [double]$EegnetKd,
        [double]$RiemannKd,
        [double]$Spatial
    )

    foreach ($directory in $Directories) {
        $csv = Join-Path $directory "riemann_guided_results.csv"
        if (-not (Test-Path -LiteralPath $csv)) {
            continue
        }
        foreach ($row in (Import-Csv -LiteralPath $csv)) {
            $datasetMatches = (
                "$($row.dataset)" -eq "bci2a" -or
                [string]::IsNullOrWhiteSpace("$($row.dataset)")
            )
            if (
                $datasetMatches -and
                [int]$row.subject -eq $Subject -and
                [int]$row.seed -eq $Seed -and
                "$($row.model_variant)" -eq "eegnetv4" -and
                (Test-FloatEqual $row.window $Window) -and
                (Test-FloatEqual $row.riemann_window 4.0) -and
                (Test-FloatEqual $row.eegnet_kd_weight $EegnetKd) -and
                (Test-FloatEqual $row.kl_weight $RiemannKd) -and
                (Test-FloatEqual $row.spatial_structure_weight $Spatial) -and
                (Test-FloatEqual $row.geometry_weight 0.0)
            ) {
                return $true
            }
        }
    }
    return $false
}

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

function Invoke-Run {
    param(
        [int]$Subject,
        [int]$Seed,
        [string]$SaveDir,
        [string[]]$SearchDirs,
        [string]$Label,
        [double]$Window,
        [double]$EegnetKd,
        [double]$RiemannKd,
        [double]$Spatial,
        [string]$TeacherPath = ""
    )

    if (Test-GuidedResult $SearchDirs $Subject $Seed $Window $EegnetKd $RiemannKd $Spatial) {
        Write-Output "Skip existing: seed=$Seed subject=$Subject $Label"
        return
    }

    Write-Output "Run: seed=$Seed subject=$Subject $Label"
    $arguments = @(
        $script,
        "--dataset", "bci2a",
        "--subject", "$Subject",
        "--window", "$Window",
        "--riemann_window", "4.0",
        "--fmin", "8",
        "--fmax", "32",
        "--tmin", "0",
        "--tmax", "4",
        "--model_variant", "eegnetv4",
        "--kernel_length", "125",
        "--eegnet_kd_weight", "$EegnetKd",
        "--kl_weight", "$RiemannKd",
        "--teacher_weight_mode", "fixed",
        "--temperature", "1.0",
        "--geometry_weight", "0",
        "--spatial_structure_weight", "$Spatial",
        "--spatial_structure_source", "filter",
        "--spatial_structure_mode", "corr",
        "--spatial_structure_trust_mode", "sample",
        "--confidence_mode", "normalized",
        "--trust_gate_mode", "confidence",
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--batch_size", "32",
        "--lr", "0.001",
        "--weight_decay", "0.0001",
        "--optimizer", "adamw",
        "--device", "cuda",
        "--save_dir", $SaveDir,
        "--patience", "$Patience"
    )
    if ($EegnetKd -gt 0) {
        if (-not (Test-Path -LiteralPath $TeacherPath)) {
            throw "Teacher checkpoint not found: $TeacherPath"
        }
        $arguments += @("--eegnet_teacher_path", $TeacherPath)
    }
    Invoke-CheckedPython $arguments
}

foreach ($seed in $Seeds) {
    $seedRoot = Join-Path $root "seed_$seed"
    $teacherDir = Join-Path $seedRoot "teacher_4s"
    $studentDir = Join-Path $seedRoot "students_2s"
    New-Item -ItemType Directory -Force -Path $teacherDir, $studentDir | Out-Null

    $searchDirs = @($studentDir, $teacherDir)
    if ($seed -eq 42) {
        $searchDirs += @(
            "results\official_eegnetv4_2s_matched_ce_baseline_seed42",
            "results\official_eegnetv4_2s_ablation_s1_spatial_filter",
            "results\official_eegnetv4_2s_riemannkd_spatial_filter_subjects1_9"
        )
    }

    foreach ($subject in $Subjects) {
        Invoke-Run $subject $seed $teacherDir $searchDirs `
            "EEGNetV4 4s teacher/upper bound" 4.0 0.0 0.0 0.0

        $teacher = Get-ChildItem -LiteralPath $teacherDir `
            -Filter "bci2a_subject_${subject}_eegnetv4_window_4_riemwin_4_seed_${seed}_eegkd_0_riekd_0_geo_0_supcon_0_*_best.pth" `
            -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $teacher) {
            throw "No EEGNetV4 teacher checkpoint found for seed=$seed subject=$subject."
        }

        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 2s CE" 2.0 0.0 0.0 0.0
        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 + EEGNet teacher KD" 2.0 0.1 0.0 0.0 $teacher.FullName
        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 + Riemann KD" 2.0 0.0 0.1 0.0
        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 + spatial only" 2.0 0.0 0.0 0.2
        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 + Riemann KD + spatial" 2.0 0.0 0.1 0.2
        Invoke-Run $subject $seed $studentDir $searchDirs `
            "EEGNetV4 + dual KD + spatial" 2.0 0.1 0.1 0.2 $teacher.FullName
    }
}

Write-Output "BCI2a EEGNetV4 strong-backbone matrix completed."
