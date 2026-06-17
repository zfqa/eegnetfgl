$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$teacherScript = "train_teacher.py"
$studentScript = "train_student_riemann_guided.py"
$teacherDir = "results\unified_teachers\bci2b\seed_42"
$studentDir = "results\bci2b_compact_2s_dual_teacher_seed42"

New-Item -ItemType Directory -Force -Path $teacherDir, $studentDir | Out-Null

foreach ($subject in 1..9) {
    $teacherPth = Join-Path $teacherDir "bci2b_subject_${subject}_teacher_best.pth"

    Write-Output "=== Subject ${subject}: train 4s compact teacher ==="
    & $python $teacherScript `
        --dataset bci2b `
        --subject $subject `
        --protocol moabb_standard `
        --model_variant compact `
        --selection_metric nll `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --seed 42 `
        --device cuda `
        --save_dir $teacherDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Compact teacher failed for subject $subject."
    }

    if (-not (Test-Path -LiteralPath $teacherPth)) {
        throw "Teacher checkpoint not found: $teacherPth"
    }

    Write-Output "=== Subject ${subject}: 2s compact CE baseline ==="
    & $python $studentScript `
        --dataset bci2b `
        --subject $subject `
        --window 2.0 `
        --riemann_window 4.0 `
        --fmin 8 `
        --fmax 32 `
        --tmin 0 `
        --tmax 4 `
        --model_variant compact `
        --eegnet_kd_weight 0 `
        --kl_weight 0 `
        --geometry_weight 0 `
        --spatial_structure_weight 0 `
        --seed 42 `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --device cuda `
        --save_dir $studentDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Compact baseline failed for subject $subject."
    }

    Write-Output "=== Subject ${subject}: 2s compact dual teacher KD ==="
    & $python $studentScript `
        --dataset bci2b `
        --subject $subject `
        --window 2.0 `
        --riemann_window 4.0 `
        --fmin 8 `
        --fmax 32 `
        --tmin 0 `
        --tmax 4 `
        --model_variant compact `
        --eegnet_teacher_path $teacherPth `
        --eegnet_kd_weight 0.4 `
        --kl_weight 0.4 `
        --temperature 1.0 `
        --geometry_weight 0 `
        --spatial_structure_weight 0 `
        --seed 42 `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --device cuda `
        --save_dir $studentDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Dual-teacher student failed for subject $subject."
    }
    Write-Output "=== Completed subject $subject ==="
}
