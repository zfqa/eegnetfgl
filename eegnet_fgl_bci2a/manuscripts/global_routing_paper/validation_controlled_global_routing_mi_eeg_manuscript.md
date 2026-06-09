# Validation-Controlled Global Routing with Neural and Riemannian Teachers for Short-Window Motor Imagery EEG Decoding

**Authors:** [Author name(s)]
**Affiliations:** [Department, Institution, City, Country]
**Corresponding author:** [Name and email]

## Abstract

Short-window motor imagery electroencephalography (MI-EEG) decoding can reduce the amount of signal required before classification, but a single temporal window and a single supervision strategy do not benefit all subjects equally. This study proposes validation-controlled global routing, a subject-level strategy-selection framework that chooses among four 2-s Compact EEGNet candidates: fixed-window cross-entropy (CE), detector-selected adaptive-window CE, fixed-window dual-teacher knowledge distillation (KD), and adaptive-window dual-teacher KD. The dual teachers comprise a frozen 4-s neural EEGNet teacher (Compact EEGNet for dataset 2a and EEGNetV4 for dataset 2b) and a 4-s Riemannian tangent-space logistic-regression classifier. Candidate selection uses only training-session validation accuracy; the evaluation session remains isolated for final testing. Subject-dependent experiments were conducted on BCI Competition IV datasets 2a and 2b using nine subjects and five random seeds. On dataset 2a, routing improved accuracy from 56.72% for fixed 2-s CE to 59.31% (+2.59 percentage points), with improvements in eight of nine subjects (paired t-test p=0.0055; Wilcoxon p=0.0078). The routed result was numerically higher than Compact EEGNet 4-s CE (59.31% versus 58.36%) but not significantly different (p=0.6466). On dataset 2b, routing increased fixed 2-s CE from 76.34% to 77.37%, but the gain was not significant (p=0.0720), and Compact EEGNet 4-s CE remained higher (78.06%). Across routed runs, the mean effective signal endpoint was 2.11 s on dataset 2a and 2.20 s on dataset 2b, compared with 4 s for the long-window reference. These results show that validation-controlled routing can accommodate subject-specific differences in temporal-window and teacher utility, with the strongest evidence on the four-class dataset 2a. The method should be interpreted as a short-window strategy selection framework rather than a universal replacement for full-window decoding.

**Keywords:** brain-computer interface; electroencephalography; motor imagery; temporal routing; knowledge distillation; Riemannian geometry; EEGNet

## 1 Introduction

Motor imagery brain-computer interfaces (BCIs) infer imagined movements from electroencephalography (EEG) and can support communication, rehabilitation, and device control. Many offline decoding studies use several seconds of post-cue EEG. Longer windows provide more temporal evidence, but they also increase the amount of signal that must be observed before a decision can be made. Short-window decoding is therefore attractive for responsive systems, although early segments may contain incomplete event-related desynchronization and synchronization patterns.

Compact convolutional models and covariance-based Riemannian classifiers offer complementary approaches to MI-EEG. EEGNet learns temporal filters and depthwise spatial filters directly from the waveform, whereas Riemannian methods represent each trial by a symmetric positive-definite channel covariance matrix and perform classification with manifold-aware distances or tangent-space features. A compact neural model is attractive at deployment, while a full-window covariance classifier can provide robust geometric information during training.

Knowledge distillation allows privileged long-window information to supervise a short-window student. However, our preliminary experiments showed substantial subject dependence. A 4-s neural or Riemannian teacher improved some subjects but caused negative transfer in others. Similarly, detector-selected temporal windows were useful for some subjects but inferior to the fixed 0-2 s segment for others. These observations suggest that forcing one temporal or supervisory strategy on every subject is poorly matched to MI-EEG heterogeneity.

We therefore formulate short-window decoding as validation-controlled strategy routing. Four complete 2-s candidate decoders are trained for each subject and seed: fixed-window CE, adaptive-window CE, fixed-window dual-teacher KD, and adaptive-window dual-teacher KD. A global router selects one candidate using only training-session validation accuracy. The router is global in the sense that it chooses one decoding strategy for a subject and training seed; it is not a per-trial mixture-of-experts gate. At test time, teachers are absent, and the temporal detector is used only when an adaptive candidate was selected.

The contributions of this work are:

- A validation-controlled global routing framework that jointly adapts temporal-window selection and the use of long-window teacher knowledge.
- A four-candidate design combining fixed or detector-selected 2-s windows with CE or dual-teacher supervision from dataset-specific 4-s neural EEGNet and Riemannian teachers.
- A leakage-controlled subject-dependent evaluation on BCI Competition IV datasets 2a and 2b, covering nine subjects and five random seeds.
- Subject-level statistical, route-frequency, long-window, and failure-mode analyses that distinguish routing error from short-window information limitations.

## 2 Materials and methods

### 2.1 Datasets and protocol

BCI Competition IV dataset 2a was accessed through MOABB as BNCI2014_001. It contains 22-channel EEG from nine subjects performing four MI classes: left hand, right hand, both feet, and tongue. Dataset 2b was accessed as BNCI2014_004 and contains left- versus right-hand MI from nine subjects using three bipolar EEG channels. Both datasets were sampled at 250 Hz.

Experiments were subject dependent and cross session. Designated training sessions were divided into stratified training and validation subsets using an 80:20 ratio. Evaluation sessions were retained as untouched test data. This split was repeated with seeds 42, 2024, 3407, 7, and 123. All model fitting, detector fitting, checkpoint selection, teacher fitting, and route selection used training-session data only.

**Table 1. Dataset characteristics and protocol.**

| Dataset | Classes | EEG channels | Sampling | Protocol |
| --- | --- | --- | --- | --- |
| BCI IV-2a / BNCI2014_001 | 4 | 22 | 250 Hz | Training session -> train/validation; evaluation session -> test |
| BCI IV-2b / BNCI2014_004 | 2 | 3 | 250 Hz | Training sessions -> train/validation; evaluation sessions -> test |

### 2.2 Preprocessing and leakage control

EEG trials were band-pass filtered from 8 to 32 Hz and cropped to 0-4 s after the cue. Per-channel standardization parameters were estimated from the training subset and applied unchanged to validation and test data. Fixed short-window models used the first 2 s. Adaptive windows always retained a 2-s duration but used a detector-selected starting point.

For dual-teacher training, the 2-s student crop and 4-s teacher input were taken from the same trial before batching. Labels and trial order were preserved. Neither teacher was queried during test inference. The test session was not used to choose a window, detector threshold, model checkpoint, or route.

### 2.3 Compact student and long-window teachers

All routed candidate classifiers used Compact EEGNet with F1=8 temporal filters, depth multiplier D=2, F2=16 pointwise filters, a 64-sample temporal kernel, ELU activations, average pooling, and dropout of 0.5. The frozen 4-s neural teacher was Compact EEGNet for dataset 2a and EEGNetV4 for dataset 2b; each teacher was trained on the same subject. The Riemannian teacher estimated 4-s trial covariance matrices with Oracle Approximating Shrinkage, projected them to the tangent space using the Riemannian metric, and fitted multinomial logistic regression. Its regularization parameter was selected on validation data from {0.01, 0.03, 0.1, 0.3, 1, 3, 10}.

The dual-teacher objective was L = CE + alpha_E KL_E + alpha_R KL_R. Here CE is cross-entropy with the true label, KL_E transfers the 4-s neural EEGNet class distribution, and KL_R transfers the 4-s Riemannian class distribution. The main setting used alpha_E=0.4, alpha_R=0.4, and temperature T=1.0.

### 2.4 Adaptive temporal-window candidates

Dataset 2a used a class-aware MI-active detector with 0.5-s windows and a 0.25-s step. Early windows were assigned a non-MI label, stable MI windows were assigned the trial class, and transition windows were excluded. Detection required a subject-specific confidence threshold and a specified number of consecutive positive windows. Threshold and consecutive-window rules had been selected from training-session validation data in the initial seed-42 pilot and were fixed before the five-seed routing experiment.

Dataset 2b used an EEGNetV4 detector trained to classify 1-s sliding windows with a 0.25-s step over the 0-2 s detection range. Confidence scores were smoothed over three neighboring windows, and the peak-confidence start was used to crop the final 2-s segment. The dataset-specific detector implementations reflect differences in channel count and task structure and are not claimed to be a single unified onset-detection algorithm.

### 2.5 Validation-controlled global routing

For each subject and seed, four complete candidates were trained: fixed CE, adaptive CE, fixed dual KD, and adaptive dual KD. Every candidate was evaluated on the same validation subset. The route with the highest validation accuracy was selected. Exact ties used a conservative priority order: fixed CE, fixed dual KD, adaptive CE, and adaptive dual KD. After selection, only the chosen candidate was scored on the evaluation session.

The route is selected once for a subject and training seed rather than for each test trial. Consequently, a fixed route deploys only Compact EEGNet. An adaptive route deploys the temporal detector followed by Compact EEGNet. Teachers remain training-only components in both cases.

### 2.6 Optimization and evaluation

Neural candidates were trained for up to 300 epochs with batch size 32, learning rate 0.001, weight decay 0.0001, and early-stopping patience of 100 epochs. Accuracy, macro-F1, and Cohen's kappa were reported. Results were first averaged over the five seeds within each subject. Paired t-tests and Wilcoxon signed-rank tests were then applied across the nine subject means. The primary comparison was global routing versus fixed 2-s Compact EEGNet CE.

Contextual baselines were evaluated under the same subject-dependent, cross-session, five-seed protocol. They comprised EEGNetV4 trained with CE, Riemannian tangent-space logistic regression using OAS covariance estimation, and CSP followed by shrinkage LDA, each evaluated with 2-s and 4-s inputs. Every reported baseline therefore contains 45 runs (nine subjects multiplied by five seeds). ShallowConvNet and DeepConvNet were not evaluated and are not included in the reported comparisons. For comparisons between routing and the six external baselines, five-seed results were first averaged within each subject, followed by paired t-tests and Wilcoxon signed-rank tests across nine subjects. Holm correction was applied separately within each dataset across the six comparisons.

For practical interpretation, the effective signal endpoint was defined as 2.0 s for a fixed route and adaptive start + 2.0 s for an adaptive route. This quantity describes the latest EEG sample used by the selected classifier; it does not include hardware, preprocessing, or neural-network execution time.

## 3 Results

### 3.1 Main result on BCI Competition IV-2a

Global routing achieved 59.31% accuracy, 58.42% macro-F1, and kappa 0.4575. Fixed 2-s CE achieved 56.72%, 55.50%, and 0.4229, respectively. The routed accuracy gain was +2.59 percentage points; eight of nine subjects improved and one declined. The difference was significant by paired t-test (p=0.0055) and Wilcoxon test (p=0.0078). If the two dataset-level primary comparisons are Holm-corrected together, the dataset 2a result remains significant (adjusted p=0.0110 for the t-test).

The fixed dual-teacher candidate reached 58.47%, confirming that privileged 4-s supervision was useful on average. Routing added a further 0.84 percentage points, but the routed-versus-fixed-dual difference was not significant (p=0.1741). The benefit of routing therefore arose from avoiding uniformly poor candidate choices and exploiting subject-specific variation, rather than from a large improvement over the strongest single candidate.

**Table 2. Candidate and routed performance.**

| Dataset | Method | Window | Accuracy | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| BCI IV-2a | Fixed CE | 0-2 s | 0.5672 | 0.5550 | 0.4229 |
| BCI IV-2a | Adaptive CE | Selected 2 s | 0.5481 | 0.5350 | 0.3974 |
| BCI IV-2a | Fixed dual KD | 0-2 s | 0.5847 | 0.5724 | 0.4463 |
| BCI IV-2a | Adaptive dual KD | Selected 2 s | 0.5353 | 0.5224 | 0.3805 |
| BCI IV-2a | Global routing | Selected route | 0.5931 | 0.5842 | 0.4575 |
| BCI IV-2b | Fixed CE | 0-2 s | 0.7634 | 0.7566 | 0.5269 |
| BCI IV-2b | Adaptive CE | Selected 2 s | 0.7722 | 0.7696 | 0.5445 |
| BCI IV-2b | Fixed dual KD | 0-2 s | 0.7677 | 0.7610 | 0.5354 |
| BCI IV-2b | Adaptive dual KD | Selected 2 s | 0.7744 | 0.7703 | 0.5488 |
| BCI IV-2b | Global routing | Selected route | 0.7737 | 0.7694 | 0.5473 |

### 3.2 Cross-dataset result on BCI Competition IV-2b

On dataset 2b, global routing reached 77.37% accuracy, 76.94% macro-F1, and kappa 0.5473. Fixed 2-s CE reached 76.34%, 75.66%, and 0.5269. The +1.02-point accuracy gain favored routing in seven of nine subjects but did not reach significance (paired t-test p=0.0720; Wilcoxon p=0.0703).

Adaptive dual KD had a slightly higher unconditionally averaged accuracy than the selected route (77.44% versus 77.37%). This illustrates validation-to-test selection noise: a route can be optimal on the available validation subset yet fail to maximize aggregate test accuracy. The external dataset therefore supports directional generalization but not a statistically reliable benefit.

### 3.3 Comparison with neural and classical baselines

On dataset 2a, global routing was numerically higher than Compact EEGNet 4-s CE (59.31% versus 58.36%; +0.96 percentage points; six wins and three losses), but the difference was not significant (paired t-test p=0.6466; Wilcoxon p=0.6523). On dataset 2b, Compact EEGNet 4-s CE remained higher (78.06% versus 77.37%; -0.69 percentage points; paired t-test p=0.4160). Global routing also remained below EEGNetV4 4-s CE on both datasets.

Among the additional five-seed baselines on dataset 2a, global routing outperformed EEGNetV4 2-s CE (59.31% versus 58.09%), Riemannian TS-LogReg 2 s (56.94%), and CSP-LDA 2 s (55.30%). The strongest overall baseline was Riemannian TS-LogReg 4 s at 65.25%, followed by CSP-LDA 4 s at 60.29% and EEGNetV4 4-s CE at 60.25%. Thus, routing was competitive among short-window methods but did not exceed the strongest full-window covariance classifier.

The central strong-backbone comparison did not reach significance: routing exceeded EEGNetV4 2-s CE by 1.23 percentage points, with seven subject-level wins and two losses (paired t-test p=0.1981; Wilcoxon p=0.2031). Routing also exceeded Riemannian TS-LogReg 2 s by 2.37 points without significance (p=0.1443), whereas its 4.01-point advantage over CSP-LDA 2 s was significant and survived Holm correction (t-test adjusted p=0.0411; Wilcoxon adjusted p=0.0234). Compared with 4-s baselines, routing was not different from CSP-LDA or EEGNetV4 and was lower than Riemannian TS-LogReg; the latter difference was significant before, but not after, six-comparison Holm correction.

On dataset 2b, EEGNetV4 2-s CE reached 77.14%, slightly below global routing at 77.37%, whereas Riemannian TS-LogReg 2 s and CSP-LDA 2 s reached 71.59% and 71.36%, respectively. EEGNetV4 4-s CE remained the strongest evaluated method at 78.95%. CSP-LDA 4 s and Riemannian TS-LogReg 4 s reached 73.03% and 72.97%.

On dataset 2b, the 0.23-point difference from EEGNetV4 2-s CE was not significant (paired t-test p=0.4885; Wilcoxon p=0.5781). Advantages over the 2-s Riemannian and CSP-LDA baselines were significant before correction but not after Holm correction. Routing exceeded CSP-LDA 4 s with Holm-adjusted Wilcoxon p=0.0469 and exceeded Riemannian TS-LogReg 4 s with Holm-adjusted t-test p=0.0345; the alternative paired test did not remain significant in either case. Routing remained below EEGNetV4 4-s CE without a significant difference (p=0.1324).

The mean effective signal endpoint of the routed strategy was 2.11 s on dataset 2a and 2.20 s on dataset 2b because fixed routes were selected in most runs. These values are substantially shorter than the 4-s reference, although they should not be interpreted as measured online decision latency.

**Table 3. Paired comparisons.**

| Dataset | Comparison | Delta acc. | W/T/L | t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | Routing vs fixed 2-s CE | +0.0259 | 8/0/1 | 0.0055 | 0.0078 |
| 2a | Routing vs fixed dual KD | +0.0084 | 6/0/3 | 0.1741 | 0.2031 |
| 2a | Routing vs Compact 4-s CE | +0.0096 | 6/0/3 | 0.6466 | 0.6523 |
| 2b | Routing vs fixed 2-s CE | +0.0102 | 7/0/2 | 0.0720 | 0.0703 |
| 2b | Routing vs fixed dual KD | +0.0059 | 6/0/3 | 0.4175 | 0.3594 |
| 2b | Routing vs Compact 4-s CE | -0.0069 | 3/0/6 | 0.4160 | 0.4961 |

**Table 4. Five-seed neural and classical baselines.**

| Dataset | Method | Window | Accuracy | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | Global routing | Selected 2 s | 0.5931 | 0.5842 | 0.4575 |
| 2a | EEGNetV4 CE | 2 s | 0.5809 | 0.5666 | 0.4412 |
| 2a | Riemann TS-LogReg | 2 s | 0.5694 | 0.5570 | 0.4259 |
| 2a | CSP-LDA | 2 s | 0.5530 | 0.5270 | 0.4040 |
| 2a | Compact EEGNet CE | 4 s | 0.5836 | 0.5749 | 0.4448 |
| 2a | EEGNetV4 CE | 4 s | 0.6025 | 0.5894 | 0.4700 |
| 2a | Riemann TS-LogReg | 4 s | 0.6525 | 0.6420 | 0.5367 |
| 2a | CSP-LDA | 4 s | 0.6029 | 0.5726 | 0.4705 |
| 2b | Global routing | Selected 2 s | 0.7737 | 0.7694 | 0.5473 |
| 2b | EEGNetV4 CE | 2 s | 0.7714 | 0.7676 | 0.5427 |
| 2b | Riemann TS-LogReg | 2 s | 0.7159 | 0.7080 | 0.4317 |
| 2b | CSP-LDA | 2 s | 0.7136 | 0.7076 | 0.4272 |
| 2b | Compact EEGNet CE | 4 s | 0.7806 | 0.7763 | 0.5612 |
| 2b | EEGNetV4 CE | 4 s | 0.7895 | 0.7852 | 0.5790 |
| 2b | Riemann TS-LogReg | 4 s | 0.7297 | 0.7189 | 0.4593 |
| 2b | CSP-LDA | 4 s | 0.7303 | 0.7197 | 0.4606 |

**Table 5. Paired comparisons with external baselines (Holm-adjusted p values in parentheses).**

| Dataset | Comparator | Delta acc. | W/T/L | t-test p (Holm) | Wilcoxon p (Holm) | dz |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | EEGNetV4 2-s CE | +0.0123 | 7/0/2 | 0.1981 (0.5943) | 0.2031 (0.8125) | 0.47 |
| 2a | Riemann TS-LogReg 2 s | +0.0237 | 6/0/3 | 0.1443 (0.5772) | 0.2383 (0.8125) | 0.54 |
| 2a | CSP-LDA 2 s | +0.0401 | 9/0/0 | 0.0069 (0.0411) | 0.0039 (0.0234) | 1.20 |
| 2a | CSP-LDA 4 s | -0.0097 | 5/0/4 | 0.7148 (1.0000) | 1.0000 (1.0000) | -0.13 |
| 2a | EEGNetV4 4-s CE | -0.0093 | 5/0/4 | 0.5901 (1.0000) | 1.0000 (1.0000) | -0.19 |
| 2a | Riemann TS-LogReg 4 s | -0.0594 | 1/0/8 | 0.0108 (0.0542) | 0.0195 (0.0977) | -1.10 |
| 2b | EEGNetV4 2-s CE | +0.0023 | 4/1/4 | 0.4885 (0.4885) | 0.5781 (0.5781) | 0.24 |
| 2b | Riemann TS-LogReg 2 s | +0.0578 | 8/0/1 | 0.0228 (0.0756) | 0.0195 (0.0781) | 0.94 |
| 2b | CSP-LDA 2 s | +0.0601 | 7/0/2 | 0.0189 (0.0756) | 0.0195 (0.0781) | 0.98 |
| 2b | CSP-LDA 4 s | +0.0433 | 8/0/1 | 0.0120 (0.0600) | 0.0078 (0.0469) | 1.08 |
| 2b | EEGNetV4 4-s CE | -0.0158 | 2/0/7 | 0.1324 (0.2648) | 0.2031 (0.4062) | -0.56 |
| 2b | Riemann TS-LogReg 4 s | +0.0440 | 8/0/1 | 0.0058 (0.0345) | 0.0117 (0.0586) | 1.24 |

### 3.4 Route usage and subject heterogeneity

Across 45 runs per dataset, fixed routes were selected in 64.4% of dataset 2a runs and 62.2% of dataset 2b runs. Dual-teacher routes were selected in 48.9% and 46.7%, respectively. No single candidate dominated, supporting the premise that both temporal-window and teacher utility were subject and seed dependent.

Exploratory test-oracle diagnostics were used only to understand failure modes. Within the evaluated candidate set, the best 2-s route still remained below 4-s CE for dataset 2a subjects 1, 2, and 3 and dataset 2b subject 9, suggesting a short-window information limitation. In contrast, dataset 2b subjects 1, 7, and 8 had a 2-s candidate capable of matching or exceeding 4-s CE, but validation routing did not consistently select it. These oracle observations were not used for model selection or headline performance.

**Table 6. Route counts.**

| Dataset | Fixed CE | Fixed dual | Adaptive CE | Adaptive dual | Fixed / adaptive |
| --- | --- | --- | --- | --- | --- |
| BCI IV-2a | 14 | 15 | 9 | 7 | 29 / 16 |
| BCI IV-2b | 17 | 11 | 7 | 10 | 28 / 17 |

## 4 Discussion

The main finding is that validation-controlled routing significantly improved Compact EEGNet short-window decoding on the four-class dataset 2a. The result was not produced by one universally superior expert. Fixed dual KD was the strongest individual candidate on dataset 2a, while adaptive candidates helped selected subjects. The router combined these complementary behaviors and avoided applying temporal adaptation or teacher supervision when validation evidence did not support them.

The result also clarifies the role of long-window teachers. Distillation was useful for several difficult subjects, but it was not uniformly beneficial. A 4-s teacher observes temporal content that may be absent from a 2-s student crop, and the semantic correspondence becomes weaker when the student window is shifted by the detector. Validation routing treats teacher knowledge as optional rather than mandatory, reducing but not eliminating negative transfer.

The comparison with 4-s CE supports a practical but cautious interpretation. On dataset 2a, the routed 2-s strategy was numerically above Compact EEGNet 4-s CE, whereas on dataset 2b it was slightly below. Neither comparison was significant. Accordingly, the evidence supports performance retention with a shorter signal segment, not general superiority over long-window decoding. A formal non-inferiority analysis with a prospectively specified margin would strengthen this claim.

Cross-dataset behavior reveals an important limitation. Dataset 2b has only two classes and three EEG channels, and the fixed 2-s baseline was already high. The smaller opportunity for improvement, together with validation selection noise, resulted in a non-significant gain. Moreover, the dataset 2a and 2b detectors use different objectives. Future work should evaluate a unified detector and an out-of-fold routing protocol before describing the system as a general temporal routing architecture.

The broader baseline matrix places the contribution more precisely. Global routing was higher than the evaluated 2-s EEGNetV4, Riemannian TS-LogReg, and CSP-LDA baselines on both datasets. However, full-window performance depended strongly on the dataset: Riemannian TS-LogReg 4 s was strongest on dataset 2a, whereas EEGNetV4 4 s was strongest on dataset 2b. The proposed contribution is therefore improved short-window strategy selection, not state-of-the-art accuracy across unrestricted window lengths.

The strongest next methodological step is out-of-fold route estimation within the training session. Candidate utility could be estimated across several internal folds, with a conservative safety threshold that activates adaptive or teacher-guided routes only when their advantage is stable. Validation-controlled spatial guidance is another candidate extension, but the current spatial-gating pilot was developed on one dataset 2b subject and is therefore not included as a main paper result.

### 4.1 Limitations

- The router selects among separately trained candidates and is not an end-to-end or per-trial mixture-of-experts gate.
- A single 20% validation split is used for route selection; comparing four candidates can amplify validation noise.
- Dataset-specific detector definitions reduce architectural uniformity across datasets.
- The effective signal endpoint is reported, but real-time wall-clock latency and asynchronous operation were not measured.
- The method significantly improved fixed 2-s CE only on dataset 2a and did not significantly outperform fixed dual KD or 4-s CE.
- Subject-dependent training limits conclusions about cross-subject transfer and calibration-free deployment.
- ShallowConvNet and DeepConvNet were not included in the completed experimental matrix; comparison with these convolutional baselines remains future work.

## 5 Conclusion

This study reframed short-window MI-EEG decoding as validation-controlled strategy routing. For each subject, the proposed framework selected among fixed or adaptive 2-s windows and CE or dual-teacher supervision. Across nine subjects and five seeds, global routing significantly improved fixed 2-s Compact EEGNet on BCI Competition IV-2a and produced a smaller, non-significant gain on IV-2b. Performance was close to Compact EEGNet 4-s CE while using a mean effective signal endpoint near 2.1-2.2 s, but the method did not significantly exceed the long-window reference. The results support validation-controlled routing as a promising way to manage subject-specific temporal and teacher heterogeneity, while motivating out-of-fold selection and unified detector designs for stronger generalization.

## Declarations

**Funding:** [Insert funding statement.]

**Competing interests:** The authors declare no competing interests.

**Ethics approval:** This study used publicly available, de-identified datasets.

**Code availability:** https://github.com/zfqa/eegnetfgl

## References

1. Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans Biomed Eng 59:920-928. https://doi.org/10.1109/TBME.2011.2172210
2. Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039
3. Blankertz B, Tomioka R, Lemm S, Kawanabe M, Muller KR (2008) Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Process Mag 25:41-56. https://doi.org/10.1109/MSP.2008.4408441
4. Chen Y, Wiesel A, Eldar YC, Hero AO (2010) Shrinkage algorithms for MMSE covariance estimation. IEEE Trans Signal Process 58:5016-5029. https://doi.org/10.1109/TSP.2010.2053029
5. Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531. https://doi.org/10.48550/arXiv.1503.02531
6. Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. J Neural Eng 15:066011. https://doi.org/10.1088/1741-2552/aadea0
7. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng 15:056013. https://doi.org/10.1088/1741-2552/aace8c
8. Lotte F, Bougrain L, Cichocki A et al (2018) A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. J Neural Eng 15:031005. https://doi.org/10.1088/1741-2552/aab2f2
9. Schirrmeister RT, Springenberg JT, Fiederer LDJ et al (2017) Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp 38:5391-5420. https://doi.org/10.1002/hbm.23730
10. Tangermann M, Muller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Front Neurosci 6:55. https://doi.org/10.3389/fnins.2012.00055
