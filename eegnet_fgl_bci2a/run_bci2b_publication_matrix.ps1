param(
    [int[]]$Seeds = @(42, 2024, 3407, 7, 123),
    [int[]]$Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$Epochs = 300,
    [int]$Patience = 100,
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
$root = "results\bci2b_paper_multiseed"

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
    $teacherDir = Join-Path $seedRoot "eegnetv4_4s_teacher"
    $studentDir = Join-Path $seedRoot "deep_models"
    New-Item -ItemType Directory -Force -Path $teacherDir, $studentDir | Out-Null

    $searchDirs = @($studentDir, $teacherDir)
    if ($seed -eq 42) {
        $searchDirs += @(
            "results\bci2b_eegnetv4_4s_teachers_seed42",
            "results\bci2b_compact_2s_dual_teacher_seed42",
            "results\bci2b_compact_2s_single_teacher_ablation_seed42",
            "results\bci2b_eegnetv4_2s_matched_ce_vs_riemann_spatial_seed42",
            "results\bci2b_compact_2s_teacher_vs_4s_seed42"
        )
    }

    foreach ($subject in $Subjects) {
        Invoke-GuidedRun $subject $seed $teacherDir $searchDirs `
            "EEGNetV4 4s teacher/upper bound" "eegnetv4" 4.0 0.0 0.0 0.0

        $teacherCandidates = @(
            Get-ChildItem -LiteralPath $teacherDir -Filter "bci2b_subject_${subject}_eegnetv4_window_4_riemwin_4_seed_${seed}_eegkd_0_riekd_0_geo_0_supcon_0_*_best.pth" -ErrorAction SilentlyContinue
        )
        if ($seed -eq 42) {
            $teacherCandidates += @(
                Get-ChildItem -LiteralPath "results\bci2b_eegnetv4_4s_teachers_seed42" `
                    -Filter "bci2b_subject_${subject}_eegnetv4_window_4_riemwin_4_seed_42_eegkd_0_riekd_0_geo_0_supcon_0_*_best.pth" `
                    -ErrorAction SilentlyContinue
            )
        }
        $teacher = $teacherCandidates | Select-Object -First 1
        if ($null -eq $teacher) {
            throw "No EEGNetV4 teacher checkpoint found for seed=$seed subject=$subject."
        }

        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Compact EEGNet 2s CE" "compact" 2.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Compact EEGNet 4s CE" "compact" 4.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "EEGNetV4 2s CE" "eegnetv4" 2.0 0.0 0.0 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "EEGNetV4 teacher only" "compact" 2.0 0.4 0.0 0.0 $teacher.FullName
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Riemann teacher only" "compact" 2.0 0.0 0.4 0.0
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Dual teacher KD" "compact" 2.0 0.4 0.4 0.0 $teacher.FullName
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Riemann spatial only" "compact" 2.0 0.0 0.0 0.1
        Invoke-GuidedRun $subject $seed $studentDir $searchDirs `
            "Riemann KD + spatial" "compact" 2.0 0.0 0.2 0.1
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
