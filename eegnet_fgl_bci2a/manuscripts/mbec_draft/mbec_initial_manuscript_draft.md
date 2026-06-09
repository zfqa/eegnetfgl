# Riemannian-Guided Knowledge Distillation and Spatial-Structure Regularization for Short-Window Motor Imagery EEG Decoding

Initial manuscript draft prepared for Medical & Biological Engineering & Computing

## Title Page

Authors: [Author Name(s)]

Affiliations: [Department, University/Institution, City, Country]

Corresponding author: [Corresponding author email; telephone; fax if required]

Word count: [to be updated after revision]

Number of figures: 3

Number of tables: 3

## Abstract

Short-window motor imagery electroencephalography (EEG) decoding is important for reducing brain-computer interface response latency, but shorter segments provide less temporal evidence than conventional 4-s trials. This study investigated whether covariance-geometry priors from 4-s EEG can improve 2-s EEGNet-based decoding without requiring long-window input at inference. Using the BNCI2014_001/BCI Competition IV-2a dataset, we evaluated subject-dependent four-class decoding across nine subjects under session-based train-test separation. A compact 2-s EEGNet student was trained with cross-entropy and dual teacher distillation from a 4-s EEGNet teacher and a 4-s Riemannian tangent-space logistic-regression teacher. Across five random seeds, Riemannian-guided dual distillation improved accuracy from 0.5672 to 0.5847 compared with the compact 2-s baseline (paired t-test p=0.0179; Wilcoxon p=0.0222). A matched EEGNetV4 analysis further showed a modest, non-significant improvement when combining Riemannian soft labels with spatial-filter structure regularization (0.6115 to 0.6181; p=0.5374). These results suggest that Riemannian geometry can provide useful supervisory information for short-window EEG decoding, although gains remain subject-dependent and require broader validation.

## Keywords

Brain-computer interface; Electroencephalography; Motor imagery; Riemannian geometry; Knowledge distillation

## 1 Introduction

Motor imagery (MI) brain-computer interfaces (BCIs) translate imagined movement-related EEG patterns into discrete control commands. For practical BCIs, reducing decision latency is clinically and ergonomically important, because long analysis windows can delay feedback and increase user fatigue. However, short-window EEG decoding is difficult: a 2-s segment contains less temporal evidence than a full 4-s motor imagery trial and may contain incomplete event-related desynchronization/synchronization patterns.

Deep convolutional models such as EEGNet have become widely used for EEG classification because their temporal convolution, depthwise spatial filtering, and separable convolution stages provide a compact architecture suited to small EEG datasets [5]. In parallel, Riemannian geometry has become a strong framework for MI-EEG decoding because covariance matrices capture cross-channel spatial relationships and can be classified on the manifold of symmetric positive definite matrices [1, 2]. These two lines of work are complementary: EEGNet learns discriminative temporal-spatial features directly from the signal, whereas Riemannian methods use covariance geometry that is often robust for MI classification.

This study asks whether 4-s Riemannian geometric knowledge can supervise a 2-s neural student so that only short-window EEG is required at inference. The proposed approach uses two forms of Riemannian guidance. First, a 4-s Riemannian tangent-space logistic-regression model produces soft-label targets for knowledge distillation. Second, the covariance-derived spatial structure is used to regularize the spatial filtering behavior of an EEGNetV4 student. The central hypothesis is that geometry extracted from full-trial covariance can guide short-window representation learning without using 4-s EEG at test time.

The main contribution is an empirical evaluation of Riemannian-guided short-window EEG decoding on the BNCI2014_001/BCI Competition IV-2a dataset. The study compares compact EEGNet baselines, single-teacher and dual-teacher distillation, Riemannian geometry/spatial ablations, EEGNetV4 baselines, Riemannian MDM/logistic-regression baselines, and CSP+LDA. The results support a modest but statistically significant improvement for compact 2-s EEGNet under multi-seed evaluation, and a smaller non-significant improvement for EEGNetV4 under matched single-seed evaluation.

## 2 Methods

## 2.1 Dataset and Experimental Protocol

Experiments used BNCI2014_001, commonly known as BCI Competition IV dataset 2a. The dataset contains four-class MI EEG recordings from nine subjects, with 22 EEG channels, 3 EOG channels, a 250-Hz sampling rate, and two sessions per subject [6]. The four classes are left hand, right hand, both feet, and tongue. This work used EEG channels only.

A subject-dependent session-based protocol was used. For each subject, the training session was split into training and validation subsets, and the evaluation session was kept as the final test set. This separation was used to avoid leakage across sessions. Standardization parameters were fitted on the training subset and then applied to validation and test data. Unless otherwise stated, 2-s students used the first 2 s of each trial, whereas 4-s teachers and long-window baselines used the full 4 s segment.

## 2.2 Preprocessing

For the main compact EEGNet experiments, EEG was band-pass filtered from 8 to 32 Hz, following a conventional sensorimotor-rhythm range. For the matched EEGNetV4 spatial-regularization analysis, the configuration inherited from the best preliminary EEGNetV4 Riemannian-guided run used a 4-38 Hz band, a temporal kernel length of 64 samples, and the Adam optimizer. These settings were matched between the EEGNetV4 baseline and the Riemannian-guided EEGNetV4 model to isolate the loss-function effect.

For Riemannian teachers and classical Riemannian baselines, trial covariance matrices were estimated with shrinkage/OAS covariance estimation and mapped to tangent-space features where logistic regression was used. Hyperparameters were selected only from training/validation data.

## 2.3 Compact Dual-Teacher Distillation

The compact student was a 2-s EEGNet trained with cross-entropy and teacher distillation. The 4-s EEGNet teacher was trained on full-window EEG from the same subject and provided neural soft-label targets. The 4-s Riemannian teacher was a tangent-space logistic-regression classifier trained from full-window covariance features and provided geometry-derived soft-label targets.

The dual-teacher loss was defined as L = L_CE + alpha_E L_KD,E + alpha_R L_KD,R, where L_CE is cross-entropy with the ground-truth label, L_KD,E is the Kullback-Leibler divergence between student probabilities and 4-s EEGNet teacher probabilities, and L_KD,R is the corresponding divergence to the Riemannian teacher probabilities. The main setting used alpha_E=0.4, alpha_R=0.4, and temperature T=1.0.

## 2.4 Riemannian Spatial-Structure Regularization for EEGNetV4

To test whether the Riemannian prior can guide a stronger neural backbone, an additional EEGNetV4 experiment was performed. The student was an EEGNetV4 model receiving only 2-s EEG. Riemannian guidance used the same-trial 4-s covariance teacher. In addition to Riemannian soft-label distillation, a spatial-structure loss constrained the depthwise spatial convolution filters of EEGNetV4. Specifically, the filter-correlation structure extracted from the depthwise spatial convolution was encouraged to match the covariance-derived Riemannian spatial structure. This setting is denoted EEGNetV4 + RiemannKD + Spatial.

The matched EEGNetV4 comparison used the same subject split, frequency range, optimizer, temporal kernel length, seed, and training code for both baseline and guided models. The baseline used kl_weight=0 and spatial_structure_weight=0. The guided model used kl_weight=0.1 and spatial_structure_weight=0.2.

## 2.5 Baselines and Ablations

The primary compact experiments compared Compact EEGNet 2s, Compact EEGNet 4s, EEGNet-teacher-only KD, Riemann-teacher-only KD, Riemann-guided dual KD, spatial-only regularization, RiemannKD + spatial, RiemannKD + geometry, and RiemannKD + geometry + spatial. The compact matrix was evaluated over nine subjects and five seeds (42, 2024, 3407, 7, and 123), producing 45 paired comparisons.

Additional seed-42 contextual baselines included EEGNetV4 2s/4s, Riemannian MDM 2s/4s, Riemannian tangent-space logistic regression 2s/4s, and CSP+LDA 2s/4s. These baselines were used to understand method strength relative to classical and stronger neural approaches, but only the compact matrix had full five-seed evaluation at the time of this draft.

## 2.6 Evaluation and Statistics

The evaluation metrics were classification accuracy, macro-averaged F1 score, and Cohen kappa. For multi-seed compact experiments, paired t-tests and Wilcoxon signed-rank tests were applied to subject-seed paired accuracies relative to the Compact EEGNet 2s baseline. For the matched EEGNetV4 analysis, paired tests were applied across the nine subjects for seed 42. Statistical analyses were performed in Python using SciPy.

## 3 Results

## 3.1 Compact EEGNet Multi-Seed Evaluation

Table 1 summarizes the main compact EEGNet results. Riemannian-guided dual KD achieved the highest mean accuracy among the compact 2-s methods, improving from 0.5672 for Compact EEGNet 2s to 0.5847. This improvement was statistically significant across 45 subject-seed pairs (paired t-test p=0.0179; Wilcoxon p=0.0222), with 29 wins, 15 losses, and 1 tie compared with the 2-s baseline.

The 2-s dual-teacher method also approached the Compact EEGNet 4s upper reference (0.5847 vs 0.5836 mean accuracy). However, the difference relative to the 4-s compact baseline should be interpreted cautiously because the 4-s baseline did not significantly outperform the 2-s compact baseline in the available paired test.

## 3.2 Strong Baseline Context

Table 2 reports seed-42 contextual baselines. The strongest single-seed baseline was Riemannian tangent-space logistic regression using 4-s EEG, with a mean accuracy of 0.6543. EEGNetV4 2s reached 0.5965, and EEGNetV4 4s reached 0.6103. These results indicate that the compact dual-teacher method improves compact short-window decoding, but it should not be claimed as state-of-the-art relative to all stronger 4-s or EEGNetV4 baselines.

## 3.3 Matched EEGNetV4 Riemannian Spatial Regularization

Table 3 provides a matched EEGNetV4 comparison using the same preprocessing and optimization settings for the baseline and the guided model. EEGNetV4 + RiemannKD + Spatial improved mean accuracy from 0.6115 to 0.6181, macro-F1 from 0.6036 to 0.6104, and kappa from 0.4820 to 0.4907. The accuracy improvement was small and not statistically significant across nine subjects (paired t-test p=0.5374; Wilcoxon p=0.4609). Subject-wise results showed five wins, three losses, and one tie.

## 4 Discussion

The results suggest that Riemannian geometry can provide useful supervisory information for short-window MI-EEG decoding, but the effect size depends strongly on the model backbone and subject. In the compact EEGNet setting, the dual-teacher design produced a statistically significant improvement over the 2-s baseline. This supports the idea that a full-window Riemannian teacher and a full-window neural teacher contain complementary information: the EEGNet teacher captures discriminative temporal-spatial patterns, while the Riemannian teacher encodes covariance geometry across EEG channels.

The EEGNetV4 analysis was more conservative. When the baseline and guided model were fully matched, RiemannKD + Spatial increased the mean accuracy by only 0.0066 and did not reach statistical significance. This result is important because it prevents overclaiming. The spatial-filter regularizer appears promising for some subjects, especially subjects 1 and 6, but it can harm others, such as subjects 7 and 8. Future work should therefore investigate subject-adaptive weighting, validation-controlled trust gates, or confidence-aware Riemannian supervision rather than applying a fixed spatial loss to every subject.

The strong performance of Riemann LogReg 4s also has methodological implications. Classical covariance-based models remain highly competitive on MI-EEG, particularly when full-window information is available. Rather than replacing these models, the present approach uses them as teachers to improve short-window neural models. This framing is important for biomedical engineering: the goal is not only maximizing offline 4-s accuracy, but also transferring reliable long-window spatial information into low-latency neural decoders.

Several limitations remain. First, the strongest compact result has full five-seed support, but the EEGNetV4 and classical baseline matrix is currently available only for seed 42. Second, the study uses a single public dataset; cross-dataset validation on additional MI corpora is needed. Third, subject-specific variability is substantial, and fixed loss weights may not be optimal. Fourth, this draft does not yet include external real-time or online BCI validation, so claims should be limited to offline subject-dependent decoding.

## 5 Conclusion

This study presents a Riemannian-guided training strategy for short-window MI-EEG decoding. In compact EEGNet experiments across nine subjects and five seeds, dual distillation from a 4-s EEGNet teacher and a 4-s Riemannian teacher significantly improved 2-s decoding accuracy. A matched EEGNetV4 experiment showed a smaller, non-significant improvement when combining Riemannian KD with spatial-filter structure regularization. Overall, the findings support Riemannian geometry as a useful supervisory prior for low-latency EEG decoding, while also showing that stronger backbones and subject variability require further validation before broad claims can be made.

## Table 1. Compact EEGNet multi-seed results on BNCI2014_001. Values are mean across nine subjects and five seeds (n=45).

| Method | n | Mean accuracy | SD accuracy | Mean F1 | Mean kappa |

| --- | --- | --- | --- | --- | --- |

| Riemann-guided dual KD 2s | 45 | 0.5847 | 0.1642 | 0.5724 | 0.4463 |

| Compact EEGNet 4s | 45 | 0.5836 | 0.1775 | 0.5749 | 0.4448 |

| EEGNet teacher KD 2s | 45 | 0.5742 | 0.1586 | 0.5620 | 0.4322 |

| RiemannKD + Geometry 2s | 45 | 0.5701 | 0.1648 | 0.5571 | 0.4269 |

| RiemannKD + Geometry + Spatial 2s | 45 | 0.5681 | 0.1702 | 0.5561 | 0.4242 |

| Compact EEGNet 2s | 45 | 0.5672 | 0.1667 | 0.5550 | 0.4229 |

| Spatial only 2s | 45 | 0.5667 | 0.1693 | 0.5549 | 0.4223 |

| Riemann teacher KD 2s | 45 | 0.5655 | 0.1645 | 0.5536 | 0.4207 |

| RiemannKD + Spatial 2s | 45 | 0.5647 | 0.1712 | 0.5528 | 0.4195 |

## Table 2. Seed-42 contextual baselines. These results are included as strength-of-baseline context and are not yet five-seed estimates.

| Method | Subjects | Seeds | Mean accuracy | Mean F1 | Mean kappa |

| --- | --- | --- | --- | --- | --- |

| Riemann LogReg 4s | 9 | 1 | 0.6543 | 0.6439 | 0.5391 |

| CSP + LDA 4s | 9 | 1 | 0.6111 | 0.5800 | 0.4815 |

| EEGNetV4 4s | 9 | 1 | 0.6103 | 0.5958 | 0.4805 |

| EEGNetV4 2s | 9 | 1 | 0.5965 | 0.5835 | 0.4619 |

| Riemann MDM 4s | 9 | 1 | 0.5922 | 0.5700 | 0.4563 |

| Riemann LogReg 2s | 9 | 1 | 0.5799 | 0.5716 | 0.4398 |

| CSP + LDA 2s | 9 | 1 | 0.5529 | 0.5226 | 0.4038 |

## Table 3. Matched EEGNetV4 analysis for the spatial-filter regularization configuration. Both rows used 2-s EEGNetV4, seed 42, 4-38 Hz filtering, temporal kernel length 64, and Adam optimization.

| Method | Subjects | Mean accuracy | Mean F1 | Mean kappa | Notes |

| --- | --- | --- | --- | --- | --- |

| EEGNetV4 CE baseline | 9 | 0.6115 | 0.6036 | 0.4820 | kl=0, spatial=0 |

| EEGNetV4 + RiemannKD + Spatial | 9 | 0.6181 | 0.6104 | 0.4907 | kl=0.1, spatial=0.2 |

| Difference | 9 | +0.0066 | +0.0068 | +0.0087 | 5 wins, 3 losses, 1 tie; t-test p=0.5374; Wilcoxon p=0.4609 |

## Acknowledgments
[Add acknowledgments, if any.]

## Declarations

Funding: [No funding was received / Add funding information.]

Competing interests: The authors declare no competing interests.

Ethics approval: This study used a publicly available, de-identified EEG dataset. No new human participant data were collected. Please verify and cite the original dataset ethics approval if required by the journal.

Consent to participate: Not applicable to the present secondary analysis; informed consent was handled by the original dataset investigators.

Data availability: The BNCI2014_001 dataset is publicly available through BNCI Horizon/MOABB. Experimental result files are available in the project repository and can be released upon acceptance.

Code availability: [Add GitHub repository URL after cleaning and documenting the code.]

Author contributions: [Add CRediT-style author contribution statements.]

## References

1. Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Transactions on Biomedical Engineering 59:920-928. https://doi.org/10.1109/TBME.2011.2172210

2. Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039

3. Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531

4. Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. Journal of Neural Engineering 15:066011. https://doi.org/10.1088/1741-2552/aadea0

5. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering 15:056013. https://doi.org/10.1088/1741-2552/aace8c

6. Tangermann M, Mueller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Frontiers in Neuroscience 6:55. https://doi.org/10.3389/fnins.2012.00055

## Figure legends

Fig. 1 Proposed Riemannian-guided short-window decoding pipeline. During training, 2-s EEG is provided to the student, while 4-s EEG from the same trial is used to train or query the Riemannian teacher and, when applicable, the 4-s EEGNet teacher. During inference, only the 2-s student is used.

Fig. 2 Subject-wise accuracy difference between Riemannian-guided dual KD and Compact EEGNet 2s across the five random seeds. Positive values indicate improved short-window decoding.

Fig. 3 Matched EEGNetV4 subject-wise differences for RiemannKD + Spatial relative to the EEGNetV4 CE baseline.
