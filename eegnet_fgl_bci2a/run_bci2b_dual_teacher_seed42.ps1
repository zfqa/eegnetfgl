$ErrorActionPreference = "Stop"

$env:MNE_DATA = "data\mne_data"
$env:MNE_DATASETS_BNCI_PATH = "data\mne_data"
$env:MNE_DONTWRITE_HOME = "true"

$python = "D:\anaconda\envs\eeg_fgl\python.exe"
$script = "train_student_riemann_guided.py"
$teacherDir = "results\bci2b_eegnetv4_4s_teachers_seed42"
$studentDir = "results\bci2b_compact_2s_dual_teacher_seed42"

foreach ($subject in 1..9) {
    Write-Output "=== Subject ${subject}: train 4s EEGNetV4 teacher ==="
    & $python $script `
        --dataset bci2b `
        --subject $subject `
        --window 4.0 `
        --riemann_window 4.0 `
        --fmin 8 `
        --fmax 32 `
        --tmin 0 `
        --tmax 4 `
        --model_variant eegnetv4 `
        --kernel_length 125 `
        --eegnet_kd_weight 0 `
        --kl_weight 0 `
        --geometry_weight 0 `
        --spatial_structure_weight 0 `
        --seed 42 `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --optimizer adamw `
        --device cuda `
        --save_dir $teacherDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "EEGNetV4 teacher failed for subject $subject."
    }

    $teacher = Get-ChildItem -LiteralPath $teacherDir -Filter "bci2b_subject_${subject}_eegnetv4_window_4_riemwin_4_seed_42_eegkd_0_riekd_0_geo_0_supcon_0_spatial_0_*_best.pth" |
        Select-Object -First 1
    if ($null -eq $teacher) {
        throw "Teacher checkpoint was not found for subject $subject."
    }

    Write-Output "=== Subject ${subject}: 2s compact CE baseline ==="
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
        --eegnet_kd_weight 0 `
        --kl_weight 0 `
        --geometry_weight 0 `
        --spatial_structure_weight 0 `
        --seed 42 `
        --epochs 300 `
        --batch_size 32 `
        --lr 0.001 `
        --weight_decay 0.0001 `
        --optimizer adamw `
        --device cuda `
        --save_dir $studentDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Compact baseline failed for subject $subject."
    }

    Write-Output "=== Subject ${subject}: 2s compact dual teacher KD ==="
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
        --save_dir $studentDir `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "Dual-teacher student failed for subject $subject."
    }
    Write-Output "=== Completed subject $subject ==="
}
