param(
    [int[]] $Subjects = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]   $Seed = 42,
    [int]   $Epochs = 300,
    [int]   $Patience = 100
)

$ErrorActionPreference = "Stop"

$python       = "D:\anaconda\envs\eeg_fgl\python.exe"
$script       = "train_student_dual_kd.py"
$teacherDir   = "results\moabb_standard_teacher_4s_subjects1_9"

# ── 1. EEGNet teacher KD only ────────────────────────────────────────────────
$eegnetSaveDir = "results\moabb_standard_eegnet_only_kd_2s_subjects1_9"
Write-Output "=== EEGNet teacher KD only (no diffusion) ==="

foreach ($subject in $Subjects) {
    $teacherPath = Join-Path $teacherDir "subject_${subject}_teacher_best.pth"
    Write-Output "`n[EEGNet-KD] Subject $subject"
    & $python $script `
        --subject $subject `
        --teacher_mode eegnet `
        --alpha1 0.4 --alpha2 0.0 `
        --temperature 1.0 `
        --diffusion_ratio 0 `
        --protocol moabb_standard `
        --seed $Seed `
        --epochs $Epochs `
        --patience $Patience `
        --eegnet_teacher_path $teacherPath `
        --rerun_existing `
        --save_dir $eegnetSaveDir
    if ($LASTEXITCODE -ne 0) {
        throw "[EEGNet-KD] Subject $subject failed with exit code $LASTEXITCODE."
    }
}

# ── 2. Riemann teacher KD only ───────────────────────────────────────────────
$riemannSaveDir = "results\moabb_standard_riemann_only_kd_2s_subjects1_9"
Write-Output "`n=== Riemann teacher KD only (no diffusion) ==="

foreach ($subject in $Subjects) {
    Write-Output "`n[Riemann-KD] Subject $subject"
    & $python $script `
        --subject $subject `
        --teacher_mode riemann `
        --alpha1 0.0 --alpha2 0.4 `
        --temperature 1.0 `
        --diffusion_ratio 0 `
        --protocol moabb_standard `
        --seed $Seed `
        --epochs $Epochs `
        --patience $Patience `
        --riemann_teacher_style paper `
        --riemann_paper_method ts_logreg `
        --riemann_cov_estimator oas `
        --riemann_alignment none `
        --riemann_metric riemann `
        --rerun_existing `
        --save_dir $riemannSaveDir
    if ($LASTEXITCODE -ne 0) {
        throw "[Riemann-KD] Subject $subject failed with exit code $LASTEXITCODE."
    }
}

Write-Output "`nAll single-teacher ablation experiments completed."
