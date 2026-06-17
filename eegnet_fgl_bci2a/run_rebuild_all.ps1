$ErrorActionPreference = "Continue"
$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"
$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$logDir = "results\rebuild_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$Seeds = @(42, 2024, 3407, 7, 123)
$Subjects = 1..9

function Run-Cmd {
    param([string]$Name, [string[]]$CmdArgs)
    $logFile = Join-Path $logDir "$Name.log"
    $errFile = Join-Path $logDir "$Name.err"
    Write-Output ">>> $Name" -NoNewline
    $proc = Start-Process -FilePath $python -ArgumentList $CmdArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        Write-Output " FAIL($($proc.ExitCode))"
    } else {
        Write-Output " OK"
    }
}

# Skip helper
function Test-Existing {
    param([string]$CsvPath, [int]$Subject, [int]$Seed, [double]$Window, [string]$Method)
    if (-not (Test-Path -LiteralPath $CsvPath)) { return $false }
    foreach ($row in (Import-Csv -LiteralPath $CsvPath)) {
        if ([int]$row.subject -eq $Subject -and [int]$row.seed -eq $Seed -and [double]$row.window -eq $Window -and "$($row.method)" -eq $Method) {
            return $true
        }
    }
    return $false
}

# ============================================================
Write-Output "`n=== PHASE 1: 2b Riemann baselines (5 seeds x 9 subj x 2 win x 2 method) ==="
foreach ($seed in $Seeds) {
    $saveDir = "results/bci2b_riemann_rebuild_seed$seed"
    foreach ($subject in $Subjects) {
        $name = "riemann_2b_s${subject}_seed${seed}"
        $csv = Join-Path $saveDir "riemann_paper_results.csv"
        # Check if already done (at least the 2s row exists)
        if (Test-Path -LiteralPath $csv) {
            $done = (Import-Csv -LiteralPath $csv | Where-Object { [int]$_.subject -eq $subject -and [int]$_.seed -eq $seed }).Count
            if ($done -ge 4) { continue }
        }
        Run-Cmd $name @(
            "train_riemann_paper.py",
            "--dataset", "bci2b", "--subject", "$subject",
            "--window", "2", "--seed", "$seed",
            "--save_dir", $saveDir,
            "--methods", "ts_logreg", "mdm",
            "--estimators", "oas", "--alignments", "none",
            "--metric", "riemann", "--calibration", "temperature",
            "--selection", "inner_cv",
            "--fmin", "8", "--fmax", "32", "--tmin", "0", "--tmax", "4",
            "--rerun_existing"
        )
        Run-Cmd "${name}_4s" @(
            "train_riemann_paper.py",
            "--dataset", "bci2b", "--subject", "$subject",
            "--window", "4", "--seed", "$seed",
            "--save_dir", $saveDir,
            "--methods", "ts_logreg", "mdm",
            "--estimators", "oas", "--alignments", "none",
            "--metric", "riemann", "--calibration", "temperature",
            "--selection", "inner_cv",
            "--fmin", "8", "--fmax", "32", "--tmin", "0", "--tmax", "4",
            "--rerun_existing"
        )
    }
}

# ============================================================
Write-Output "`n=== PHASE 2: 2a Riemann baselines (5 seeds x 9 subj x 2 win x 2 method) ==="
foreach ($seed in $Seeds) {
    $saveDir = "results/riemann_paper_rebuild_seed$seed"
    foreach ($subject in $Subjects) {
        $name = "riemann_2a_s${subject}_seed${seed}"
        $saveDir = "results/riemann_paper_rebuild_seed$seed"
        Run-Cmd $name @(
            "train_riemann_paper.py",
            "--dataset", "bci2a", "--subject", "$subject",
            "--window", "2", "--seed", "$seed",
            "--save_dir", $saveDir,
            "--methods", "ts_logreg", "mdm",
            "--estimators", "oas", "--alignments", "none",
            "--metric", "riemann", "--calibration", "temperature",
            "--selection", "inner_cv",
            "--fmin", "8", "--fmax", "32", "--tmin", "0", "--tmax", "4",
            "--rerun_existing"
        )
        Run-Cmd "${name}_4s" @(
            "train_riemann_paper.py",
            "--dataset", "bci2a", "--subject", "$subject",
            "--window", "4", "--seed", "$seed",
            "--save_dir", $saveDir,
            "--methods", "ts_logreg", "mdm",
            "--estimators", "oas", "--alignments", "none",
            "--metric", "riemann", "--calibration", "temperature",
            "--selection", "inner_cv",
            "--fmin", "8", "--fmax", "32", "--tmin", "0", "--tmax", "4",
            "--rerun_existing"
        )
    }
}

# ============================================================
Write-Output "`n=== PHASE 3: 2b weight-matched KD ablation ==="
foreach ($seed in $Seeds) {
    foreach ($subject in $Subjects) {
        foreach ($config in @("eegnet_08", "riemann_08", "dual_02")) {
            $name = "wmkd_2b_s${subject}_${config}_seed${seed}"
            $saveDir = "results/weight_matched_kd_rebuild/bci2b/seed_${seed}/subject_${subject}"
            $csv = Join-Path $saveDir "riemann_guided_results.csv"
            if (Test-Path -LiteralPath $csv) {
                $existing = Import-Csv -LiteralPath $csv | Where-Object {
                    "$($_.subject)" -eq "$subject" -and "$($_.seed)" -eq "$seed"
                }
                if ($existing) { continue }
            }
            Run-Cmd $name @(
                "run_weight_matched_kd_ablation.py",
                "--datasets", "bci2b",
                "--subjects", "$subject",
                "--seeds", "$seed",
                "--configs", $config,
                "--epochs", "300", "--patience", "100",
                "--device", "cuda",
                "--root_dir", "results/weight_matched_kd_rebuild"
            )
        }
    }
}

# ============================================================
Write-Output "`n=== PHASE 4: 2a weight-matched KD ablation ==="
foreach ($seed in $Seeds) {
    foreach ($subject in $Subjects) {
        foreach ($config in @("eegnet_08", "riemann_08", "dual_02")) {
            $name = "wmkd_2a_s${subject}_${config}_seed${seed}"
            $saveDir = "results/weight_matched_kd_rebuild/bci2a/seed_${seed}/subject_${subject}"
            $csv = Join-Path $saveDir "dual_kd_results.csv"
            if (Test-Path -LiteralPath $csv) {
                $existing = Import-Csv -LiteralPath $csv | Where-Object {
                    "$($_.subject)" -eq "$subject" -and "$($_.seed)" -eq "$seed"
                }
                if ($existing) { continue }
            }
            Run-Cmd $name @(
                "run_weight_matched_kd_ablation.py",
                "--datasets", "bci2a",
                "--subjects", "$subject",
                "--seeds", "$seed",
                "--configs", $config,
                "--epochs", "300", "--patience", "100",
                "--device", "cuda",
                "--root_dir", "results/weight_matched_kd_rebuild"
            )
        }
    }
}

Write-Output "`n=== ALL REBUILDS COMPLETE ==="
