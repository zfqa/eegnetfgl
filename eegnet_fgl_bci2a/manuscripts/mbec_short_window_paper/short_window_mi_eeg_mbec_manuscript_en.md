# Short-window motor-imagery EEG decoding with training-time privileged information and validation-controlled routing

## Abstract

Motor-imagery EEG (MI-EEG) decoders often use several seconds of post-cue data, which improves evidence accumulation but delays a decision. We investigated whether a student decoder restricted to the first 2 s can benefit from information available only during training. The framework used 4-s neural and Riemannian teachers in the four-class BCI Competition IV-2a setting and Riemannian soft-label and spatial supervision in the binary BCI Competition IV-2b setting. Candidate short-window strategies were selected using only a validation partition of the training session; the independent evaluation session was never used for teacher fitting, model selection, or routing. Across nine participants and five random seeds, validation-controlled routing on IV-2a achieved 59.31% accuracy, compared with 56.72% for fixed 2-s cross-entropy training (mean paired difference 2.59 percentage points; paired t-test p=0.0055; Wilcoxon p=0.0078). On IV-2b, adaptive Riemannian knowledge distillation with spatial supervision achieved the highest short-window mean accuracy (77.12%), whereas routing reached 77.01% and fixed 2-s cross-entropy reached 76.34%. The smaller IV-2b gain indicates that the benefit depends on task complexity and available spatial information. Teachers and Riemannian targets were removed at inference, leaving only a 2-s Compact EEGNet and, when selected, a temporal detector. These results support training-time privileged information and validation-controlled strategy selection as a route to earlier MI-EEG decisions, but not as a universal replacement for complete-window decoding.

**Keywords:** motor imagery; electroencephalography; brain-computer interface; short-window decoding; knowledge distillation; Riemannian geometry; EEGNet

## 1 Introduction

Motor-imagery brain-computer interfaces (BCIs) infer an intended movement from EEG and can support communication, rehabilitation, and device control. A practical decoder must balance accuracy against the amount of signal that must be acquired before a decision. Longer post-cue windows provide more temporal evidence, but they also impose an unavoidable waiting period. This makes short-window decoding a biomedical engineering problem rather than merely a smaller-input classification problem.

Compact convolutional networks and Riemannian covariance methods provide complementary views of MI-EEG. EEGNet learns temporal and spatial filters directly from waveform segments, whereas Riemannian approaches represent each trial by a channel covariance matrix and exploit its geometry. Knowledge distillation further provides a way to expose a short-window model to training-time information that will be unavailable at deployment. However, a single teacher or a single temporal policy can improve some participants while degrading others.

We therefore formulate early MI-EEG decoding as a training-time privileged-information problem followed by validation-controlled strategy selection. A 2-s student is the only classifier used during deployment. During training, 4-s neural and Riemannian teachers can supply class-distribution targets; for IV-2b, the Riemannian covariance structure additionally constrains the student spatial representation. On IV-2a, four candidate strategies are compared per participant and seed, and a validation set from the training session selects the candidate tested on the held-out evaluation session.

The study makes three bounded contributions. First, it demonstrates that validation-controlled routing significantly improves 2-s decoding on the more difficult four-class, 22-channel IV-2a task. Second, it shows that Riemannian spatial supervision is a more suitable short-window training prior than neural dual-teacher distillation on the three-channel IV-2b task, although the gain is small. Third, it quantifies the position of the proposed short-window methods relative to reproduced classical, compact, shallow, and deep neural baselines. The results are intended to support earlier decisions under a 2-s signal budget, not to claim that 2 s universally outperforms 4-s decoding.

## 2 Materials and methods

### 2.1 Datasets and cross-session protocol

| Dataset | Classes | EEG channels | Sampling rate | Outer protocol |
| --- | ---: | ---: | ---: | --- |
| BCI Competition IV-2a / BNCI2014_001 | 4 | 22 | 250 Hz | Training session -> train/validation; evaluation session -> test |
| BCI Competition IV-2b / BNCI2014_004 | 2 | 3 | 250 Hz | Training session -> train/validation; evaluation session -> test |

Both datasets were obtained through MOABB. For every participant, the training session was stratified into a training subset and a validation subset (80/20), while the evaluation session remained an independent test set. Five seeds (42, 2024, 3407, 7, and 123) controlled the training/validation split and neural optimization. All preprocessing parameters, teacher fitting, checkpoint selection, and route selection used only the training session.

### 2.2 Signal preprocessing and short-window task

Signals were band-pass filtered from 8 to 32 Hz and cropped to 0-4 s after the cue. Per-channel standardization was fitted on the training subset and applied unchanged to validation and test data. The fixed student input was the real 0-2 s segment. The 4-s signal was available only to training-time teachers and reference models. Student and teacher inputs for a batch always originated from the same trial, preserving labels and sample order.

### 2.3 Training-time privileged information

All student classifiers used Compact EEGNet. On IV-2a, a frozen 4-s Compact EEGNet teacher and a 4-s Riemannian teacher produced soft class targets. The latter used OAS covariance estimation, tangent-space mapping at the Riemannian mean, and logistic regression with the regularization constant selected within the training session. The dual-teacher objective was the sum of cross-entropy and neural/Riemannian Kullback-Leibler terms.

On IV-2b, the neural-Riemannian dual-teacher formulation was not retained as the main configuration because it did not consistently improve fixed 2-s cross-entropy. Instead, a 4-s Riemannian teacher supplied a soft-label term and its covariance-derived structure regularized the pre-depthwise spatial representation of the 2-s student. The loss was L = CE + lambda_R KL_R + lambda_S L_spatial, with lambda_R=0.1 and lambda_S=0.2 in the fixed main configuration. Figure 1 summarizes the separation between training-time privileged information and deployment-time computation.

### 2.4 Validation-controlled routing

For IV-2a, the candidate set comprised fixed 2-s cross-entropy, adaptive-window cross-entropy, fixed dual-teacher distillation, and adaptive dual-teacher distillation. Each candidate was evaluated on the same validation subset. The candidate with the highest validation accuracy was selected, with a conservative tie order of fixed CE, fixed dual teacher, adaptive CE, and adaptive dual teacher. The selected candidate was then evaluated once on the independent evaluation session. This is participant-level routing, not trial-level mixture-of-experts inference.

For IV-2b, fixed CE, adaptive CE, fixed Riemann KD plus spatial supervision, and adaptive Riemann KD plus spatial supervision were compared diagnostically. The detector used a 1-s sliding window with a 0.25-s step within the first 2 s. It selected a 2-s segment after confidence smoothing. Teachers and covariance targets were never called at inference.

### 2.5 Metrics and statistics

Accuracy, macro-F1, and Cohen's kappa were reported. For primary seed-level summaries, each seed was first averaged across nine participants, followed by the mean and standard deviation over five seeds. For the IV-2a primary comparison, per-participant values were first averaged over seeds and then compared with paired t-tests and Wilcoxon signed-rank tests. The fixed-versus-adaptive IV-2b Riemannian spatial comparisons are interpreted descriptively because corresponding paired tests have not yet been recomputed under the final spatial-supervision matrix.

## 3 Results

### 3.1 Validation-controlled routing improved short-window decoding on IV-2a

The principal result was observed on IV-2a (Figure 2a and Table 2). Fixed 2-s CE achieved 56.72% accuracy, whereas validation-controlled routing achieved 59.31%, an absolute gain of 2.59 percentage points. Routing improved eight of nine participants after seed averaging (8/0/1 wins/ties/losses). The difference from fixed CE was significant (paired t-test p=0.0055; Wilcoxon p=0.0078). Fixed dual-teacher training achieved 58.47%, but routing was not significantly higher than this best fixed candidate (p=0.1741). Thus, the evidence supports the value of selecting among short-window strategies across participants, rather than a claim that one candidate dominates every participant.

**Table 2 | Main short-window results.** Values with +/- report mean +/- standard deviation across five seeds after averaging each seed across nine participants. Entries without variability were retained from previously completed reference summaries.

| Dataset | Method | Input | Accuracy | Macro-F1 | Kappa |
| --- | --- | ---: | ---: | ---: | ---: |
| BCI IV-2a | Fixed 2-s CE | 2 s | 0.5672 +/- 0.0093 | 0.5550 | 0.4229 |
| BCI IV-2a | Fixed 2-s dual teacher | 2 s | 0.5847 +/- 0.0125 | 0.5724 | 0.4463 |
| BCI IV-2a | Validation-controlled routing | 2 s | 0.5931 +/- 0.0114 | 0.5842 | 0.4575 |
| BCI IV-2a | Compact EEGNet CE | 4 s | 0.5836 | 0.5749 | 0.4448 |
| BCI IV-2b | Fixed 2-s CE | 2 s | 0.7634 +/- 0.0120 | 0.7567 | 0.5268 |
| BCI IV-2b | Fixed Riemann KD + spatial | 2 s | 0.7660 +/- 0.0142 | 0.7627 | 0.5319 |
| BCI IV-2b | Adaptive CE | 2 s | 0.7675 +/- 0.0031 | 0.7637 | 0.5349 |
| BCI IV-2b | Adaptive Riemann KD + spatial | 2 s | 0.7712 +/- 0.0152 | 0.7681 | 0.5424 |
| BCI IV-2b | Validation-controlled routing | 2 s | 0.7701 +/- 0.0103 | 0.7665 | 0.5402 |
| BCI IV-2b | Compact EEGNet CE | 4 s | 0.7806 | 0.7763 | 0.5612 |

### 3.2 IV-2b showed a smaller, task-dependent short-window benefit

The pattern changed on IV-2b (Figure 2b). Fixed 2-s CE achieved 76.34% accuracy. Fixed Riemann KD plus spatial supervision increased the mean to 76.60%, while adaptive Riemann KD plus spatial supervision reached 77.12%. Validation-controlled routing reached 77.01%, slightly below the best individual candidate but with lower reported seed-level variability. These differences are modest compared with IV-2a. Because IV-2b is a binary task with three channels, the room for additional covariance-based spatial information and teacher complementarity is more limited. The results therefore support a small cross-dataset benefit rather than a universal superiority claim.

### 3.3 The proposed short-window methods were competitive with reproduced baselines

Figure 3 and Table 3 place the 2-s methods against reproduced classical and neural baselines. On IV-2a, routing exceeded ShallowConvNet (51.87%), DeepConvNet (48.63%), CSP-LDA (55.30%), FBCSP-LDA (56.38%), Riemannian tangent-space logistic regression (56.94%), and EEGNetV4 CE (58.09%). On IV-2b, adaptive Riemann KD plus spatial supervision (77.12%) was close to EEGNetV4 CE (77.14%) and exceeded the shallow/deep 2-s baselines (74.77% and 74.87%). The ShallowConvNet and DeepConvNet comparisons are descriptive because paired tests against the proposed methods have not yet been added.

**Table 3 | Reproduced 2-s external baselines.**

| Dataset | 2-s comparator | Accuracy | Macro-F1 | Kappa |
| --- | --- | ---: | ---: | ---: |
| BCI IV-2a | ShallowConvNet | 0.5187 +/- 0.0026 | 0.5113 | 0.3582 |
| BCI IV-2a | DeepConvNet | 0.4863 +/- 0.0131 | 0.4553 | 0.3151 |
| BCI IV-2a | CSP-LDA | 0.5530 | 0.5270 | 0.4040 |
| BCI IV-2a | FBCSP-LDA | 0.5638 | 0.5517 | 0.4184 |
| BCI IV-2a | Riemann TS-LogReg | 0.5694 | 0.5570 | 0.4259 |
| BCI IV-2a | EEGNetV4 CE | 0.5809 | 0.5666 | 0.4412 |
| BCI IV-2a | Validation-controlled routing | 0.5931 +/- 0.0114 | 0.5842 | 0.4575 |
| BCI IV-2b | ShallowConvNet | 0.7477 +/- 0.0073 | 0.7443 | 0.4954 |
| BCI IV-2b | DeepConvNet | 0.7487 +/- 0.0118 | 0.7429 | 0.4973 |
| BCI IV-2b | CSP-LDA | 0.7136 | 0.7076 | 0.4272 |
| BCI IV-2b | FBCSP-LDA | 0.7258 | 0.7227 | 0.4515 |
| BCI IV-2b | Riemann TS-LogReg | 0.7159 | 0.708 | 0.4317 |
| BCI IV-2b | EEGNetV4 CE | 0.7714 | 0.7676 | 0.5427 |
| BCI IV-2b | Adaptive Riemann KD + spatial | 0.7712 +/- 0.0152 | 0.7681 | 0.5424 |

### 3.4 Complete-window results remain important context

Four-second reference models generally retained an advantage on the strongest complete-window comparisons (Table 4). For example, EEGNetV4 reached 60.25% on IV-2a and 78.95% on IV-2b, while Riemannian TS-LogReg reached 65.25% on IV-2a. The short-window goal is therefore not to claim a universal replacement of 4-s decoding; it is to recover useful performance under a fixed 2-s evidence budget.

**Table 4 | Complete-window reference context. These models use 4 s of EEG and are not direct latency-matched comparators.**

| Dataset | 4-s reference | Accuracy | Macro-F1 | Kappa |
| --- | --- | ---: | ---: | ---: |
| BCI IV-2a | ShallowConvNet | 0.5038 | 0.4923 | 0.3384 |
| BCI IV-2a | DeepConvNet | 0.4710 | 0.4372 | 0.2947 |
| BCI IV-2a | Compact EEGNet CE | 0.5836 | 0.5749 | 0.4448 |
| BCI IV-2a | EEGNetV4 CE | 0.6025 | 0.5894 | 0.4700 |
| BCI IV-2a | Riemann TS-LogReg | 0.6525 | 0.6420 | 0.5367 |
| BCI IV-2a | CSP-LDA | 0.6029 | 0.5726 | 0.4705 |
| BCI IV-2a | FBCSP-LDA | 0.6296 | 0.6154 | 0.5061 |
| BCI IV-2b | ShallowConvNet | 0.7666 | 0.7619 | 0.5332 |
| BCI IV-2b | DeepConvNet | 0.7741 | 0.7639 | 0.5482 |
| BCI IV-2b | Compact EEGNet CE | 0.7806 | 0.7763 | 0.5612 |
| BCI IV-2b | EEGNetV4 CE | 0.7895 | 0.7852 | 0.5790 |
| BCI IV-2b | Riemann TS-LogReg | 0.7297 | 0.7189 | 0.4593 |
| BCI IV-2b | CSP-LDA | 0.7303 | 0.7197 | 0.4606 |
| BCI IV-2b | FBCSP-LDA | 0.7587 | 0.7540 | 0.5173 |

### 3.5 Riemannian soft-label and spatial supervision were complementary at moderate auxiliary weight

The IV-2b auxiliary-loss ablation is shown in Figure 4 and Table 5. At a total auxiliary weight of 0.20, the joint Riemann KD plus spatial configuration (0.10+0.10) reached 76.48%, exceeding Riemann KD alone (76.08%) and spatial supervision alone (76.14%). A lower-weight joint configuration (0.05+0.05) reached 75.89%. These values are participant-level summaries after averaging five seeds; their standard deviations reflect participant heterogeneity and should not be compared directly with the seed-level standard deviations in Table 2. The ablation suggests complementarity at moderate weights, but it is descriptive without a matched paired test against CE.

**Table 5 | IV-2b Riemannian KD and spatial-loss ablation at 8-32 Hz.**

| Configuration | Total auxiliary weight | Accuracy | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: | ---: |
| Riemann KD + spatial (0.05 + 0.05) | 0.10 | 0.7589 +/- 0.1420 | 0.7525 +/- 0.1494 | 0.5177 +/- 0.2840 |
| Riemann KD + spatial (0.10 + 0.10) | 0.20 | 0.7648 +/- 0.1354 | 0.7617 +/- 0.1373 | 0.5296 +/- 0.2709 |
| Riemann KD only (0.20 + 0.00) | 0.20 | 0.7608 +/- 0.1373 | 0.7543 +/- 0.1458 | 0.5216 +/- 0.2745 |
| Spatial only (0.00 + 0.20) | 0.20 | 0.7614 +/- 0.1421 | 0.7551 +/- 0.1499 | 0.5229 +/- 0.2841 |

## 4 Discussion

The central engineering result is that a 2-s MI-EEG decoder can be improved without carrying a 4-s teacher into deployment. On IV-2a, training-time long-window information and validation-controlled routing increased short-window accuracy relative to a fixed 2-s CE student. This is relevant when an interface is constrained by the duration of EEG evidence available before issuing a decision.

The IV-2a and IV-2b contrast is informative. IV-2a has four classes and 22 channels, providing a richer spatial covariance structure and a more difficult class boundary. In that setting, combining neural and Riemannian information and selecting a participant-specific route was beneficial. IV-2b is a binary three-channel task with a higher short-window baseline. Its small gain from Riemannian spatial supervision is consistent with a reduced opportunity for additional spatial structure to change the decision boundary. This difference should be interpreted as task dependence, not as a failure to reproduce the IV-2a result.

The method has important boundaries. First, the routing result uses a single internal validation split per seed. The independent evaluation session was protected from selection, but candidate selection may still be sensitive to how the training session is partitioned. An out-of-fold or inner K-fold routing study is the most important next validation. Second, short-window methods did not exceed the strongest 4-s references. Third, the study is participant-dependent and does not establish cross-participant transfer or asynchronous online performance. These constraints motivate cautious use of the term earlier decision rather than universal early decoding.

## 5 Conclusion

This study frames short-window MI-EEG decoding as an early-decision problem with training-time privileged information. A 2-s student benefited from validation-controlled strategy selection on the four-class IV-2a dataset, where routing significantly improved accuracy over fixed CE. On IV-2b, Riemannian spatial supervision produced only a small positive effect, demonstrating that the usefulness of long-window guidance depends on task complexity and channel availability. The inference pipeline retains only the short-window student and optional detector. The evidence supports a practical route toward earlier MI-EEG decisions while preserving the distinction between latency-matched short-window decoding and complete-window reference performance.

## Declarations

**Data availability.** BCI Competition IV-2a and IV-2b are publicly accessible through MOABB as BNCI2014_001 and BNCI2014_004.

**Code availability.** The training and analysis code will be released in a public repository upon acceptance. The release should include exact environment versions, result CSV files, figure source data, and commands used for every reported matrix.

**Competing interests.** The authors declare no competing interests.

## References

参考文献保留原始英文书目信息。

1. Altaheri H, Muhammad G, Alsulaiman M et al (2023) Deep learning techniques for classification of electroencephalogram (EEG) motor imagery (MI) signals: a review. Neural Comput Appl 35:14681-14722. https://doi.org/10.1007/s00521-023-08542-7
2. Ang KK, Guan C (2018) EEG-based strategies for stroke rehabilitation. In: Nam CS, Nijholt A, Lotte F (eds) Brain-computer interfaces handbook. CRC Press, pp 553-574
3. Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans Biomed Eng 59:920-928. https://doi.org/10.1109/TBME.2011.2172210
4. Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039
5. Blankertz B, Tomioka R, Lemm S, Kawanabe M, Muller KR (2008) Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Process Mag 25:41-56. https://doi.org/10.1109/MSP.2008.4408441
6. Brunner C, Leeb R, Muller-Putz G, Schlogl A, Pfurtscheller G (2010) BCI Competition 2008: Graz data set A. Institute for Knowledge Discovery, Graz University of Technology
7. Chen Y, Wiesel A, Eldar YC, Hero AO (2010) Shrinkage algorithms for MMSE covariance estimation. IEEE Trans Signal Process 58:5016-5029. https://doi.org/10.1109/TSP.2010.2053029
8. Congedo M, Barachant A, Bhatia R (2017) Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. BCI 4:1-39. https://doi.org/10.1080/2326263X.2017.1297192
9. Cui J, Lan Z, Liu Y et al (2022) A compact and interpretable convolutional neural network for cross-subject driver drowsiness recognition from single-channel EEG. IEEE Trans Neural Syst Rehabil Eng 30:286-296. https://doi.org/10.1109/TNSRE.2022.3147419
10. Ding Y, Robinson N, Zeng Q et al (2022) TSception: a deep learning framework for emotion detection using EEG. In: Proc Int Joint Conf Neural Networks (IJCNN), pp 1-8. https://doi.org/10.1109/IJCNN55064.2022.9967692
11. Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531. https://doi.org/10.48550/arXiv.1503.02531
12. Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. J Neural Eng 15:066011. https://doi.org/10.1088/1741-2552/aadea0
13. Kalunga EK, Chevallier S, Barre Q, Djouani K, Monacelli E, Hamam Y (2016) Online SSVEP-based BCI using Riemannian geometry. Neurocomputing 191:55-68. https://doi.org/10.1016/j.neucom.2016.01.007
14. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng 15:056013. https://doi.org/10.1088/1741-2552/aace8c
15. Lotte F, Bougrain L, Cichocki A et al (2018) A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. J Neural Eng 15:031005. https://doi.org/10.1088/1741-2552/aab2f2
16. Lotte F, Jeunet C (2015) Towards improved BCI spellers using a co-adaptive calibration of the P300 speller with RSVP and visual feedback. J Neural Eng 12:066007
17. Mane R, Chew E, Chua K et al (2021) FBCNet: a multi-view convolutional neural network for brain-computer interface. arXiv:2104.01233
18. Meisheri H, Ramrao N, Mitra S, Saha S (2022) Common spatial pattern for EEG based brain-computer interface: an implementation review. In: Proc Int Conf Comput Intell, pp 1-6
19. Pfurtscheller G, Neuper C (2001) Motor imagery and direct brain-computer communication. Proc IEEE 89:1123-1134. https://doi.org/10.1109/5.939829
20. Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J (2019) Deep learning-based electroencephalography analysis: a systematic review. J Neural Eng 16:051001. https://doi.org/10.1088/1741-2552/ab260c
21. Schirrmeister RT, Springenberg JT, Fiederer LDJ et al (2017) Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp 38:5391-5420. https://doi.org/10.1002/hbm.23730
22. Song Y, Zheng Q, Liu B, Gao X (2023) EEG conformer: convolutional transformer for EEG decoding and visualization. IEEE Trans Neural Syst Rehabil Eng 31:710-719. https://doi.org/10.1109/TNSRE.2022.3230250
23. Tangermann M, Muller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Front Neurosci 6:55. https://doi.org/10.3389/fnins.2012.00055
24. Tang J, Liu C, Li Q (2020) EEG feature distillation for motor imagery classification. In: Proc IEEE Int Conf Bioinformatics and Biomedicine (BIBM), pp 1680-1683
25. Wang J, Li S, Li R, Shi Y (2022) EEG motor imagery classification with multi-view convolutional neural network. Biomed Signal Process Control 77:103778
26. Wei Q, Wang Y, Gao X, Gao S (2022) Amplitude and phase coupling in steady-state motion visual evoked potentials for BCI. J Neural Eng 19:026002
27. Yger F, Berar M, Lotte F (2016) Riemannian approaches in brain-computer interfaces: a review. IEEE Trans Neural Syst Rehabil Eng 25:1753-1762. https://doi.org/10.1109/TNSRE.2016.2627016
28. Zhang L, Wang J, Chen B et al (2023) Temporal convolutional network for motor imagery EEG decoding with attention mechanism. Biomed Signal Process Control 81:104432
29. Zhang X, Yao L, Wang X, Monaghan J, McAlpine D (2022) A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers. J Neural Eng 18:031002
30. Zuo C, Jin J, Yin Y et al (2023) Novel hybrid deep learning model for motor imagery EEG classification based on capsule network. Cogn Neurodyn 17:107-121
