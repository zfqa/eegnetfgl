# Dual-Teacher Knowledge Distillation for Compact Short-Window Motor Imagery EEG Decoding

**Authors:** [Author name(s)]
**Affiliations:** [Department, Institution, City, Country]
**Corresponding author:** [Name and email]

## Abstract

Short analysis windows can reduce the response latency of motor imagery (MI) brain-computer interfaces, but they provide less temporal evidence than full-trial electroencephalography (EEG). This study investigated whether knowledge extracted from 4-s EEG can improve a compact classifier that receives only the first 2 s at inference. A 2-s Compact EEGNet student was jointly supervised by the ground-truth labels, a frozen 4-s EEGNet teacher, and a 4-s Riemannian tangent-space logistic-regression teacher. Subject-dependent cross-session experiments were conducted on BCI Competition IV datasets 2a and 2b using nine subjects and five random seeds. Training-session data were separated into training and validation subsets, whereas the evaluation sessions were retained for final testing. On dataset 2a, dual-teacher distillation increased mean accuracy from 56.72% to 58.47% (+1.75 percentage points), with improvements in seven of nine subjects (paired t-test p=0.0122; Wilcoxon p=0.0156; Cohen dz=1.07). The advantage did not remain significant after Holm correction across all compact-model comparisons. The same method produced a smaller, non-significant gain on dataset 2b (76.35% to 76.77%, p=0.4574). Moreover, applying dual and spatial Riemannian guidance to EEGNetV4 yielded only a 0.24-point gain on dataset 2a. These findings show that long-window neural and covariance-geometric teachers can improve a capacity-limited short-window student, but the effect is backbone- and dataset-dependent. The proposed method is therefore best interpreted as a compact low-latency decoding strategy rather than a universal accuracy improvement or a state-of-the-art full-window classifier.

**Keywords:** brain-computer interface; electroencephalography; motor imagery; knowledge distillation; Riemannian geometry; short-window decoding

## 1 Introduction

Motor imagery brain-computer interfaces (BCIs) infer imagined movements from electroencephalography and can provide communication, rehabilitation, and control channels without overt muscular activity. In offline studies, classification is often performed on several seconds of post-cue EEG. In an interactive system, however, a long observation interval delays feedback. Reducing the analysis window is therefore attractive, but the shorter segment may contain only a partial event-related desynchronization or synchronization response and is more vulnerable to non-stationarity and noise.

EEGNet is a compact convolutional architecture that combines temporal filtering, depthwise spatial filtering, and separable convolutions. Its low parameter count makes it appealing for subject-dependent EEG datasets, where the number of trials is limited. Riemannian methods offer a complementary view: each trial is represented by a symmetric positive-definite channel covariance matrix, and discrimination is performed using manifold-aware distances or tangent-space features. These methods are particularly relevant to MI because covariance captures cross-channel spatial organization in sensorimotor rhythms.

Knowledge distillation provides a mechanism for transferring information from a teacher to a smaller or more constrained student. Most EEG distillation studies focus on model compression or teacher-student architectures using the same input duration. A less explored setting is temporal privilege: a teacher receives a longer segment during training, while the student is restricted to an earlier segment during both training and testing. This setting is clinically and computationally meaningful because privileged full-window information is available offline but not required during deployment.

We study whether two heterogeneous 4-s teachers can improve a 2-s Compact EEGNet student. The first teacher is an EEGNet classifier that provides neural soft labels. The second teacher is a Riemannian tangent-space logistic-regression classifier that provides covariance-geometric soft labels. The student is trained with cross-entropy and two Kullback-Leibler divergence terms, but only the student is retained at inference. We further test Riemannian spatial-filter regularization on EEGNetV4, compare with classical covariance and CSP baselines, and repeat the evaluation on BCI Competition IV dataset 2b.

The contributions of this work are:

- A dual-teacher training framework that transfers same-trial 4-s neural and Riemannian soft labels to a Compact EEGNet using only the first 2 s at inference.
- A leakage-controlled, subject-dependent cross-session evaluation over nine subjects and five random seeds on BCI Competition IV-2a and IV-2b.
- Ablations covering single teachers, dual teachers, Riemannian spatial regularization, a stronger EEGNetV4 backbone, and classical CSP/Riemannian baselines.
- An explicit analysis of where the method succeeds and fails, including multiple-comparison correction, class-wise confusion, teacher-quality correlations, and cross-dataset generalization.

## 2 Materials and methods

### 2.1 Datasets

BCI Competition IV dataset 2a was accessed through MOABB as BNCI2014_001. It contains EEG from nine subjects performing four MI classes: left hand, right hand, both feet, and tongue. The experiments used the 22 EEG channels sampled at 250 Hz. Dataset 2b was accessed as BNCI2014_004 and contains left- versus right-hand MI from nine subjects using three bipolar EEG channels at 250 Hz. Both datasets provide temporally separated training and evaluation recordings, enabling cross-session assessment.

### 2.2 Preprocessing and leakage control

MOABB MotorImagery was used to load 0-4 s after-cue epochs with an 8-32 Hz band-pass filter. The analysis was subject dependent. For each subject and random seed, the designated training sessions were split into 80% training and 20% validation data using stratification. The designated evaluation sessions were never used for fitting, hyperparameter selection, teacher training, or normalization.

Per-channel standardization parameters were estimated only from the training subset and then applied to validation and test trials. The 2-s input contained the first 500 samples of each trial, whereas the 4-s input contained 1000 samples. The paired 2-s and 4-s tensors were cropped from the same trial before batching, preserving labels and sample order. Consequently, every teacher prediction used for distillation corresponded to the same trial as the student input.

### 2.3 Compact EEGNet student and neural teacher

The Compact EEGNet used F1=8 temporal filters, depth multiplier D=2, F2=16 pointwise filters, a 64-sample temporal kernel, ELU activations, average pooling, and dropout of 0.5. The 2-s student and 4-s EEGNet teacher shared the same compact architecture but had input-dependent classifier dimensions. The 4-s teacher was trained with cross-entropy on the subject's training data, selected by validation performance, frozen, and evaluated only to produce training/validation soft targets. It was not called during student test inference.

### 2.4 Riemannian teacher

For the Riemannian teacher, covariance matrices were estimated from raw 4-s training trials using Oracle Approximating Shrinkage. Symmetric positive-definite matrices were projected to the tangent space with the Riemannian metric and classified using multinomial logistic regression. The regularization parameter C was chosen from {0.01, 0.03, 0.1, 0.3, 1, 3, 10} using only the validation subset. No alignment was used in the final paper configuration. Class probabilities from the selected model served as geometry-derived soft labels.

### 2.5 Dual-teacher objective

Let z_s denote the 2-s student logits, z_e the frozen 4-s EEGNet teacher logits, and z_r the logits corresponding to the Riemannian teacher probabilities. The training objective was

L = CE(z_s, y) + alpha_E T^2 KL[p_E^T || p_S^T] + alpha_R T^2 KL[p_R^T || p_S^T],

where p^T=softmax(z/T). The main Compact EEGNet experiment used alpha_E=0.4, alpha_R=0.4, and T=1.0. Single-teacher ablations set one of the two teacher weights to zero. The test-time model consisted only of the 2-s student.

### 2.6 Strong-backbone and spatial-structure analysis

To determine whether the benefit persisted with a stronger implementation, we repeated the experiment using EEGNetV4 with exact same-padding, Keras-like batch normalization, and max-norm constraints. The main strong-backbone configuration used F1=8, D=2, F2=16, a 125-sample temporal kernel, and dropout of 0.5. Riemannian spatial regularization matched the off-diagonal correlation structure of the depthwise spatial-filter weights to a confidence-weighted average covariance-correlation target derived from 4-s trials. The dual KD plus spatial setting used neural KD weight 0.1, Riemannian KD weight 0.1, and spatial weight 0.2.

### 2.7 Training and model selection

Neural models were trained for up to 300 epochs with batch size 32, learning rate 0.001, weight decay 0.0001, and early stopping patience of 100 epochs. Compact dual-teacher experiments used AdamW. Each experiment was repeated with seeds 42, 2024, 3407, 7, and 123. Checkpoints were selected using validation accuracy; the evaluation session was scored once after model selection.

### 2.8 Baselines, metrics, and statistical analysis

Baselines included Compact EEGNet and EEGNetV4 at 2 s and 4 s, single-teacher KD, Riemannian spatial-only and combined losses, minimum-distance-to-mean (MDM), Riemannian tangent-space logistic regression, and common spatial patterns with linear discriminant analysis (CSP-LDA). Accuracy, macro-F1, and Cohen's kappa were reported.

For inference-oriented performance estimates, results were first averaged over the five seeds within each subject. Paired t-tests and Wilcoxon signed-rank tests were then performed over the nine subject means. Cohen's dz quantified the paired effect size. Holm correction was applied within each family of comparisons. Teacher-quality associations were assessed with Pearson and Spearman correlations.

## 3 Results

### 3.1 Main BCI Competition IV-2a result

Across 45 runs, Compact EEGNet 2s achieved 56.72% accuracy, 55.50% macro-F1, and kappa 0.4229. Dual-teacher KD increased these values to 58.47%, 57.24%, and 0.4463, respectively. On the nine seed-averaged subject scores, the accuracy difference was +1.75 percentage points; seven subjects improved, one was tied, and one declined. The unadjusted paired t-test (p=0.0122) and Wilcoxon test (p=0.0156) were significant, with Cohen's dz=1.07. After Holm correction across the six compact-model comparisons, the p-values were 0.0734 and 0.0938.

The dual-teacher 2-s mean was close to the Compact EEGNet 4-s reference (58.47% versus 58.36%). However, the 4-s compact model itself did not significantly outperform its 2-s counterpart across subjects. This comparison therefore suggests that the student recovered useful full-window supervisory information, but it does not establish equivalence to all long-window decoders.

### 3.2 Ablation and strong-backbone results

The neural teacher alone increased Compact EEGNet accuracy to 57.42%, whereas the Riemannian teacher alone reached 56.55%. Spatial-only and Riemannian KD plus spatial variants did not improve the compact baseline. The best result therefore required the combination of neural and Riemannian soft-label supervision rather than either teacher in isolation.

With EEGNetV4, the 2-s cross-entropy baseline reached 58.09% and the 4-s model reached 60.25%. Dual KD plus spatial regularization achieved 58.33%, an increase of only 0.24 percentage points over EEGNetV4 2s (five wins and four losses; paired t-test p=0.7570). Thus, the main improvement was specific to the smaller Compact EEGNet student and was largely absorbed by the stronger backbone.

### 3.3 Classical baselines

Riemannian tangent-space logistic regression with 4-s input was the strongest BCI IV-2a method in the evaluated matrix, obtaining 65.25% mean accuracy. CSP-LDA 4s and Riemannian MDM 4s achieved 60.29% and 59.48%, respectively. At 2 s, tangent-space logistic regression reached 56.94%, close to the Compact EEGNet baseline but below the dual-teacher student. The proposed method should therefore be interpreted as improving a compact short-window neural decoder, not as outperforming the strongest full-window classical pipeline.

### 3.4 Cross-dataset evaluation on BCI Competition IV-2b

On the binary BCI IV-2b dataset, Compact EEGNet 2s already achieved 76.35% accuracy. Dual-teacher KD increased accuracy to 76.77% (+0.42 percentage points), with six subject-level wins and three losses, but the difference was not significant (paired t-test p=0.4574; Wilcoxon p=0.4258). EEGNetV4 2s and 4s achieved 77.14% and 78.95%, respectively. The weaker relative benefit on dataset 2b suggests that the teacher signal is less useful when the short-window binary task is already comparatively easy.

### 3.5 Class-wise errors and teacher quality

Checkpoint-level re-inference reproduced the aggregate BCI IV-2a result. Dual-teacher training increased right-hand recall from 62.4% to 65.9% and feet recall from 41.3% to 44.4%; left-hand recall changed from 62.1% to 62.5%, while tongue recall remained 61.1%. The persistent difficulty of the feet class indicates that the improvement was not merely a uniform confidence shift.

Teacher validation accuracy was not a reliable predictor of Compact EEGNet improvement. At the subject-mean level, Pearson correlations were -0.119 for the EEGNet teacher (p=0.760) and -0.300 for the Riemannian teacher (p=0.433). This finding argues against a simple global trust rule based only on teacher validation accuracy and motivates future sample- or subject-adaptive gating.

## 4 Discussion

The central result is that two heterogeneous long-window teachers can improve a capacity-limited short-window neural decoder. The 4-s EEGNet teacher supplies a smooth discriminative class distribution learned directly from the waveform, while the 4-s Riemannian teacher encodes covariance geometry across channels. Neither teacher was sufficient alone in the compact ablation, but their combination yielded the highest 2-s compact-model accuracy. This pattern is consistent with complementary supervision rather than simple duplication of the ground-truth label.

The gain is practically relevant but modest in absolute terms. A 1.75-point mean increase can matter in a compact low-latency decoder, particularly because inference does not require either teacher or the second half of the trial. At the same time, the result must be interpreted with statistical care. The primary unadjusted paired comparison was significant and showed a large standardized paired effect, but it did not survive Holm correction across all compact ablations. The study therefore provides encouraging evidence rather than definitive proof of a universal benefit.

Backbone dependence is one of the most informative findings. EEGNetV4 improved the 2-s cross-entropy baseline relative to Compact EEGNet, yet Riemannian and dual guidance added almost nothing. A stronger model may already learn much of the available short-window spatial structure, reducing the marginal information supplied by the teachers. Alternatively, fixed KD and spatial weights may over-constrain a higher-capacity network. This result supports adaptive transfer rather than applying the same loss to every backbone.

Dataset dependence was similarly clear. BCI IV-2b is a two-class task with only three EEG channels, and its 2-s baseline was already high. The small, non-significant gain suggests a ceiling effect and reduced opportunity for a 22-channel covariance teacher to contribute. In contrast, the four-class IV-2a task contains a more complex decision boundary and richer spatial covariance structure. The method's most plausible use case is therefore a constrained short-window decoder on a difficult multi-class task rather than all MI-EEG settings.

The strong 4-s tangent-space logistic-regression result also deserves emphasis. Classical Riemannian approaches remain competitive and should not be treated only as weak baselines. Here, their value is twofold: they provide an interpretable benchmark and a source of privileged geometric supervision. Future work could improve transfer by learning a validation-controlled trust gate, calibrating teacher probabilities, or activating Riemannian losses only when teacher uncertainty and student uncertainty indicate useful complementary information.

### 4.1 Limitations

- The evaluation is offline and subject dependent; no asynchronous or online BCI experiment was performed.
- The main effect is modest and does not remain significant after Holm correction over all compact-model comparisons.
- The method improves Compact EEGNet on IV-2a but not EEGNetV4, indicating backbone dependence.
- Cross-dataset improvement on IV-2b is small and non-significant.
- Fixed teacher weights were used; teacher validation accuracy alone did not identify subjects that would benefit.
- The strongest full-window Riemannian baseline remained more accurate than the proposed 2-s neural student.

## 5 Conclusion

This study evaluated dual-teacher knowledge distillation from 4-s neural and Riemannian teachers to a 2-s Compact EEGNet student. Across nine subjects and five seeds on BCI Competition IV-2a, the method improved short-window accuracy by 1.75 percentage points and benefited seven subjects, while requiring only the 2-s student at inference. The benefit was not reproduced to the same extent with EEGNetV4 or on BCI IV-2b. The evidence therefore supports dual-teacher transfer as a targeted strategy for compact, difficult short-window MI decoding, while highlighting the need for adaptive teacher trust and broader validation.

## Declarations

**Funding:** [Insert funding statement.]

**Competing interests:** The authors declare no competing interests.

**Ethics approval:** This study used publicly available, de-identified datasets and collected no new participant data.

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
