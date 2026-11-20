# EDGCN supplied-code baseline:2a S1

Purpose: validate the author-supplied MODEL before any4s->2s distillation.
Reference target is Table1 S01=90.28%, NOT the prose's inconsistent97.57%
(Table1 gives97.57% to S03). This runner does NOT promise score reproduction.

## Evidence hierarchy and provenance

Unmodified user-supplied mmc1.py/mmc2.py are retained under
references/edgcn_supplement. The model port uses mmc2 as the structural source,
including where code contradicts the paper figure:40/60/80 kernels, Ghost
1x1->3x1, ghost width11, pool5/6, d_model32, input PSD width88, time_len33,
raw fusion adjacency as Chebyshev support, SUM of T/S outputs, per-node Linear,
then node-mean logits. NO Flatten head, no separate concatenated classifier,
no extra Laplacian normalization, no forced train-reference SASE graph.

Training objective uses mmc1's actual code: CE +.1 times the sum of two WHOLE
parameter-bank cosine-orthogonality losses. This differs from the paper's
per-order/trainable-coefficient description. It is not silently changed.

mmc1 cannot run as provided:6-argument forward is called with2 arguments;
private interaug/scheduler/test imports are absent; save_test_accuracy is not
defined. The port supplies a plain PyTorch loop, NOT a claim to reproduce the
missing Lightning launcher. The originals are archived, never imported.

## Known protocol vs local assumptions

Paper:22 channels,250Hz,0.5-38Hz,-0.5..4s,1125 points; train-session source and
independent evaluation session; Adam lr.0009, batch64. MOABB may return1126
points including the right endpoint; only that extra endpoint is removed.
Cache must contain data/MNE-bnci-data/~bci/database/001-2014/A01T.mat/A01E.mat.

MISSING author inputs: PSD generation, interpolation masks/downsampling,
PLV/local graph/global mask helper, full split/seed/epoch/selection config.
The source model accepts these as EXTERNAL tensors and does not define them.
Therefore the executable profile is explicitly LOCAL_ADAPTER, not exact author
preprocessing. Its assumptions are saved in run_config.json:

- train-session80/20 stratified holdout by seed42; best validation accuracy,
  then lower CE NLL, with default300 epochs/patience100. These are local choices.
  mmc1's max_epochs=1000 is a constructor default, not proof of the published run.
- MOABB offline zero-phase filtering. Signal units remain loader microvolts.
  No Z-score by default; optional --standardize fits training statistics ONLY.
- Four illustrative PSD bands0.5-4,4-8,8-13,13-38 ->22*4=88 features/window;
  half-second Hann periodogram, half-overlap, density normalization, band mean
  and log1p. Exact original-grid2x interpolation; training-only10% midpoint
  masking; downsample every second point. Actual author bands/settings unknown.
  Override through --psd_bands (8 numbers), --psd_window_s,
  --mask_probability and --psd_transform only in a new output directory.
- Broadband analytic-signal PLV, local grouping by standard electrode rows,
  global threshold.5. Graphs are averaged over each batch INSIDE author SASE,
  also during validation/test. Thus predictions ARE batch-composition dependent;
  batch64 and ordered eval batches are fixed, including the smaller final batch.
  This source behavior is not disguised as per-trial independent inference.
- No interaug, no scheduler; actual launcher flags were not supplied.
- No teacher/student, no KD, no relation target.4.5s full-model classification.

Model parameter keys, seeded initialization and train/eval predictions are tested
against passive class definitions loaded from the supplied mmc2 file. This checks
local code equivalence, not equality to an unpublished training/preprocessing run.
The exact supplied-file SHA256 hashes are stored with each experiment.

## Server: first fit/validate only

New entry/model/reference files are required; older distillation scripts cannot
substitute for this1125-point baseline. Use a new output root.

```bash
(
  set -e
  cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
    run_edgcn_supplement_baseline.py --stage fit \
    --subject 1 --seed 42 --data_dir data --device cuda \
    --epochs 300 --patience 100 --batch_size 64 --lr 0.0009 \
    --output_dir results/edgcn_supplement_s1_local_adapter \
    > logs/edgcn_supplement_s1_fit.out 2>&1 &
  echo "PID: $!"
)
```

Inspect history.csv and validation_results.csv before the explicit test stage:

```bash
cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/eeg_fgl/bin/python -u \
  run_edgcn_supplement_baseline.py --stage test \
  --subject 1 --seed 42 --data_dir data --device cuda \
  --epochs 300 --patience 100 --batch_size 64 --lr 0.0009 \
  --output_dir results/edgcn_supplement_s1_local_adapter
```

The locked checkpoint records selection epoch, data signature, normalization,
config and source hashes. Settings cannot change across stages. Completed fits
cannot be overwritten. Interrupted fits restart from seeded initialization;
there is no mid-epoch resume. Test predictions and test_results.csv are saved
only in the test stage. Do not tune repeatedly against the known S1 test session.

If scores differ from90.28%, first report the missing-author-helper limitation;
do not declare the paper invalid or select a lucky seed as a reproduction.
