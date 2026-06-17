param(
    [int[]]$Seeds = @(42, 2024, 3407, 7, 123),
    [int[]]$Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$Epochs = 300,
    [int]$Patience = 100,
    [string]$Root = "results\bci2b_paper_multiseed_unified",
    [string]$TeacherRoot = "results\unified_teachers\bci2b",
    [switch]$IncludeSpatialAblations,
    [switch]$SkipClassicalBaselines
)

$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$guidedScript = "train_student_riemann_guided.py"
$riemannScript = "train_riemann_paper.py"
$cspScript = "train_csp_baseline.py"
$root = $Root

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
        [string]$ModelVariant,
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
            if (
                "$($row.dataset)" -eq "bci2b" -and
                [int]$row.subject -eq $Subject -and
                [int]$row.seed -eq $Seed -and
                "$($row.model_variant)" -eq $ModelVariant -and
                (Test-FloatEqual $row.window $Window) -and
                (Test-FloatEqual $row.riemann_window 4.0) -and
                (Test-FloatEqual $row.eegnet_kd_weight $EegnetKd) -and
                (Test-FloatEqual $row.kl_weight $RiemannKd) -and
                (Test-FloatEqual $row.spatial_structure_weight $Spatial) -and
                (Test-FloatEqual $row.geometry_weight 0.0) -and
                "$($row.teacher_calibration)" -eq "temperature" -and
                "$($row.confidence_mode)" -eq "none" -and
                "$($row.riemann_cov_estimator)" -eq "oas" -and
                "$($row.riemann_selection)" -eq "inner_cv" -and
                [int]$row.riemann_selection_folds -eq 5
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

function Invoke-GuidedRun {
    param(
        [int]$Subject,
        [int]$Seed,
        [string]$SaveDir,
        [string[]]$SearchDirs,
        [string]$Label,
        [string]$ModelVariant,
        [double]$Window,
        [double]$EegnetKd,
        [double]$RiemannKd,
        [double]$Spatial,
        [string]$TeacherPath = ""
    )

    if (Test-GuidedResult $SearchDirs $Subject $Seed $ModelVariant $Window $EegnetKd $RiemannKd $Spatial) {
        Write-Output "Skip existing: seed=$Seed subject=$Subject $Label"
        return
    }

    Write-Output "Run: seed=$Seed subject=$Subject $Label"
    $arguments = @(
        $guidedScript,
        "--dataset", "bci2b",
        "--subject", "$Subject",
        "--window", "$Window",
        "--riemann_window", "4.0",
        "--fmin", "8",
        "--fmax", "32",
        "--tmin", "0",
        "--tmax", "4",
        "--model_variant", $ModelVariant,
        "--eegnet_kd_weight", "$EegnetKd",
        "--kl_weight", "$RiemannKd",
        "--teacher_weight_mode", "fixed",
        "--temperature", "1.0",
        "--geometry_weight", "0",
        "--spatial_structure_weight", "$Spatial",
        "--spatial_structure_source", "filter",
        "--spatial_structure_mode", "corr",
        "--spatial_structure_trust_mode", "sample",
        "--confidence_mode", "none",
        "--teacher_calibration", "temperature",
        "--riemann_cov_estimator", "oas",
        "--riemann_alignment", "none",
        "--riemann_metric", "riemann",
        "--riemann_method", "ts_logreg",
        "--riemann_selection", "inner_cv",
        "--riemann_selection_folds", "5",
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
    if ($ModelVariant -eq "eegnetv4") {
        $arguments += @("--kernel_length", "125")
    }
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
    $teacherDir = Join-Path $seedRoot "compact_4s_upper_bound"
    $studentDir = Join-Path $seedRoot "deep_models"
    New-Item -ItemType Directory -Force -Path $teacherDir, $studentDir | Out-Null

    $searchDirs = @($studentDir, $teacherDir)

    foreach ($subject in $Subjects) {
        Invoke-GuidedRun $subject $seed $teacherDir $searchDirs `
            "Compact 4s teacher/upper bound" "compact" 4.0 0.0 0.0 0.0

        $teacher = Join-Path (Join-Path $TeacherRoot "seed_$seed") "bci2b_subject_${subject}_teacher_best.pth"
        if (-not (Test-Path -LiteralPath $teacher)) {
            throw "Validated compact teacher not found: $teacher. Run run_unified_teacher_matrix.py first."
        }

        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Compact EEGNet 2s CE" "compact" 2.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Compact EEGNet 4s CE" "compact" 4.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "EEGNetV4 2s CE" "eegnetv4" 2.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Compact EEGNet teacher only" "compact" 2.0 0.4 0.0 0.0 $teacher
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Riemann teacher only" "compact" 2.0 0.0 0.4 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Dual teacher KD" "compact" 2.0 0.4 0.4 0.0 $teacher
        if ($IncludeSpatialAblations) {
            Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
                "Riemann spatial only" "compact" 2.0 0.0 0.0 0.1
            Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
                "Riemann KD + spatial" "compact" 2.0 0.0 0.2 0.1
        }
    }

    if (-not $SkipClassicalBaselines) {
        $riemannDir = Join-Path $seedRoot "riemann"
        $cspDir = Join-Path $seedRoot "csp"
        foreach ($subject in $Subjects) {
            foreach ($window in @(2.0, 4.0)) {
                Invoke-CheckedPython @(
                    $riemannScript,
                    "--dataset", "bci2b",
                    "--subject", "$subject",
                    "--window", "$window",
                    "--seed", "$seed",
                    "--save_dir", $riemannDir,
                    "--methods", "mdm", "ts_logreg",
                    "--estimators", "oas",
                    "--alignments", "none",
                    "--selection", "inner_cv",
                    "--selection_folds", "5",
                    "--calibration", "temperature",
                    "--metric", "riemann",
                    "--fmin", "8",
                    "--fmax", "32",
                    "--tmin", "0",
                    "--tmax", "4"
                )
                Invoke-CheckedPython @(
                    $cspScript,
                    "--dataset", "bci2b",
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
}

Write-Output "BCI2b publication matrix completed."
