from argparse import Namespace
from pathlib import Path

from run_teacher_window_matched_dual_kd import build_student_command


def make_args() -> Namespace:
    return Namespace(
        dataset="bci2a",
        preprocessing_mode="moabb_zero_phase",
        causal_filter_order=4,
        data_dir="data",
        batch_size=32,
        lr=0.001,
        weight_decay=0.0001,
        device="cuda",
        alpha_eegnet=0.4,
        alpha_riemann=0.4,
        temperature=1.0,
        riemann_paper_method="ts_logreg",
        riemann_cov_estimator="oas",
        riemann_alignment="none",
        riemann_metric="riemann",
        riemann_selection="inner_cv",
        riemann_selection_folds=5,
        riemann_c_grid="0.01,0.1,1",
        epochs_student=300,
        patience_student=100,
        rerun_existing=False,
    )


def test_four_second_arm_uses_both_four_second_teachers():
    command = build_student_command(
        make_args(),
        subject=3,
        seed=42,
        arm="teacher_4s",
        teacher_path=Path("teacher4.pth"),
        teacher_window=4.0,
        save_dir=Path("results"),
    )
    joined = " ".join(command)

    assert "--teacher_window 4.0" in joined
    assert "--teacher_mode dual" in joined
    assert "--alpha1 0.4" in joined
    assert "--alpha2 0.4" in joined
    assert "--eegnet_teacher_path teacher4.pth" in joined
    assert "--riemann_paper_method ts_logreg" in joined


def test_ce_arm_disables_both_teachers():
    command = build_student_command(
        make_args(),
        subject=1,
        seed=7,
        arm="ce",
        teacher_path=None,
        teacher_window=2.0,
        save_dir=Path("results"),
    )
    joined = " ".join(command)

    assert "--teacher_mode none" in joined
    assert "--alpha1 0.0" in joined
    assert "--alpha2 0.0" in joined
    assert "--eegnet_teacher_path" not in joined
