$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$script = "train_student_riemann_guided.py"
$teacherDir = "results\unified_teachers\bci2b\seed_42"
$saveDir = "results\bci2b_compact_2s_dual_teacher_val_trust_seed42"

foreach ($subject in 1..9) {
    $teacher = Get-ChildItem -LiteralPath $teacherDir -Filter "bci2b_subject_${subject}_compact_*_best.pth" |
        Select-Object -First 1
    if ($null -eq $teacher) {
        throw "Teacher checkpoint was not found for subject $subject in $teacherDir."
    }

    Write-Output "=== Subject ${subject}: validation-trusted dual teacher KD ==="
    & $python $script `
        --dataset bci2b `
        --subject $subject `
        --window 2.0 `
        --riemann_window 4.0 `
        --fmin 8 `
        --fmax 32 `
        --tmin 0 `
        --tmax 4 `
        --model_variant compact `
        --eegnet_teacher_path $teacher.FullName `
        --eegnet_kd_weight 0.4 `
        --kl_weight 0.4 `
        --teacher_weight_mode val_excess_accuracy `
        --temperature 1.0 `
        --geometry_weight 0 `
        --spatial_structure_weight 0 `
        --confidence_mode normalized `
        --trust_gate_mode confidence `
        --seed 42 `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --optimizer adamw `
        --device cuda `
        --save_dir $saveDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Validation-trusted dual teacher failed for subject $subject."
    }
    Write-Output "=== Completed subject $subject ==="
}
