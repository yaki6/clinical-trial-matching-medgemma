# Deep Research Report: MedGemma 4B Multimodal Benchmarks and Medical VQA Evaluation

**Date**: 2026-02-23
**Researcher**: deep-research-agent
**Repositories Analyzed**: google-deepmind/gemma, microsoft/LLaVA-Med, MMMU-Benchmark/MMMU, Project-MONAI/MONAI, EleutherAI/lm-evaluation-harness, mahmoodlab/CONCH, mahmoodlab/UNI, MIT-LCP/mimic-code, google-research/google-research, huggingface/evaluate
**Web Sources**: MedGemma Technical Report (arXiv:2507.05201), HuggingFace model cards, CVPR 2024 OmniMedVQA paper
**Total Research Rounds**: 5 rounds across 4 topics (20 Deepwiki queries + 6 web searches)

---

## Executive Summary

MedGemma 4B is Google's medical vision-language model built on Gemma 3 4B with a specialized MedSigLIP vision encoder. Its official technical report (arXiv:2507.05201, July 2025) evaluates it across 22+ datasets spanning 5 task types and 6 imaging modalities. The two primary VQA benchmarks Google uses are SLAKE (English subset) and VQA-RAD, measured with tokenized F1 as the primary metric rather than simple accuracy. For radiology report generation, the standard metric is RadGraph F1, not ROUGE-L. PathVQA is notably absent from MedGemma's official evaluation but is a widely used public benchmark (available on HuggingFace as `flaviagiammarino/path-vqa`). The most practically actionable benchmarks for this project are: SLAKE English (HuggingFace: `Voxel51/SLAKE`), VQA-RAD (HuggingFace: `flaviagiammarino/vqa-rad`), MMMU Health and Medicine subcategory (HuggingFace: `MMMU/MMMU_Pro`), and MIMIC-CXR with RadGraph F1 evaluation.

---

## Research Objectives

1. What benchmarks did Google officially use to evaluate MedGemma 4B multimodal capabilities?
2. What public medical VQA datasets are commonly used and where can they be accessed?
3. What metrics are standard for medical imaging AI evaluation beyond ROUGE-L?
4. Are there established radiology report generation benchmarks with proper evaluation frameworks?

---

## Detailed Findings

### Opening Item 1: Official MedGemma 4B Benchmarks from Google

#### Round 1: Surface Exploration

**Questions Asked**:
- What benchmarks did Google use to evaluate MedGemma 4B multimodal capabilities?
- What medical VQA datasets does MedGemma 4B support or was evaluated on?
- What fine-tuning recipes and evaluation frameworks does the official repository provide?

**Key Discoveries**:
- The official MedGemma repository (`google-health/medgemma`) was not indexed by DeepWiki. The technical report (arXiv:2507.05201) is the authoritative source.
- Google evaluated MedGemma across 22+ datasets, 5 task types, 6 imaging modalities.
- The Google Vertex AI samples repo confirms MedGemma exists as a Model Garden offering but has no dedicated evaluation notebooks — only general Gemma/PaliGemma evaluation patterns using `lm-evaluation-harness`.

**Initial Gaps**: Need exact benchmark numbers, dataset splits, and which metrics are primary vs. secondary.

#### Round 2: Deep Dive

**Questions Asked**: Fetched full technical report via web; searched HuggingFace model card.

**Key Discoveries — Complete Benchmark Table from Technical Report**:

##### Task 1: Medical Text QA (MedGemma 4B IT)
| Benchmark | Test Size | Metric | MedGemma 4B | Gemma 3 4B |
|-----------|-----------|--------|-------------|------------|
| MedQA | 1,273 | Accuracy | 64.4% | — |
| MedMCQA | 4,183 | Accuracy | 55.7% | — |
| PubMedQA | 500 | Accuracy | 73.4% | — |
| MMLU Anatomy | — | Accuracy | 59.3% | — |
| MMLU Clinical Knowledge | — | Accuracy | 71.3% | — |
| MMLU College Biology | — | Accuracy | 70.8% | — |
| MMLU College Medicine | — | Accuracy | 65.3% | — |
| MMLU Medical Genetics | — | Accuracy | 83.0% | — |
| MMLU Professional Medicine | — | Accuracy | 76.8% | — |
| MMLU Virology | — | Accuracy | 53.0% | — |
| AfriMed-QA | 25 | Accuracy | 52.0% | — |
| MedXpertQA (OOD) | 2,450 | Accuracy | multimodal: 24.4% | — |

##### Task 2: Medical Image Classification (MedGemma 4B)
| Dataset | Images | Conditions | Metric | MedGemma 4B | Gemma 3 4B |
|---------|--------|-----------|--------|-------------|------------|
| MIMIC-CXR (Med-Gemini split) | 1,532 | 5 CXR | Macro F1 | 88.9% | 81.1% |
| MIMIC-CXR (MAIRA split) | 2,461 | 5 CXR | Macro F1 | 40.5% | — |
| CheXpert (OOD) | 668 | 5 CXR | Macro F1 | 48.1% | 31.2% |
| CXR14 (OOD) | 1,962 | 5 CXR | Macro F1 | 50.1% | — |
| PathMCQA (histopathology) | 450 | internal | Accuracy | 69.8% | 37.1% |
| US-Derm MCQA (dermatology) | 1,996 | 136 skin cond. | Accuracy | 71.8% | 42.6% |
| EyePACS (ophthalmology) | 3,161 | DR grading | Accuracy | 64.9% | — |

##### Task 3: Medical VQA (MedGemma 4B IT)
| Dataset | Images | Metric | MedGemma 4B | Gemma 3 4B |
|---------|--------|--------|-------------|------------|
| SLAKE English | 1,061 | Overall token F1 | 72.3% | 40.2% |
| SLAKE English | 1,061 | Open-ended token recall | 63.3% | — |
| SLAKE English | 1,061 | Closed-ended accuracy | 87.6% | — |
| VQA-RAD | 2,248 | Overall token F1 | 49.9% | 33.6% |
| VQA-RAD | 2,248 | Closed Q&A accuracy | 69.1% | — |
| MedXpertQA (OOD) | 2,000 | Accuracy | 24.4% | — |

**Critical split note**: VQA-RAD uses Yang et al. (2024) custom splits, NOT the original dataset splits, to avoid train/test image contamination.

##### Task 4: CXR Report Generation (MedGemma 4B PT)
| Dataset | Cases | Metric | MedGemma 4B PT | MedVersa (SOTA) | PaliGemma 2 3B |
|---------|-------|--------|----------------|-----------------|----------------|
| MIMIC-CXR | 912 | RadGraph F1 | 29.5% | 30.0% | 28.8% |

The pretrained model (not instruction-tuned) is used because RadGraph F1 is sensitive to reporting style.

Human evaluation: 81% of generated reports resulted in same or superior clinical decisions vs. original radiologist reports.

##### Task 5: Medical Agentic Behavior (MedGemma 27B only)
| Dataset | Cases | Metric | MedGemma 27B |
|---------|-------|--------|--------------|
| AgentClinic-MedQA | 215 | Accuracy | 56.2% |
| AgentClinic-MIMIC-IV | 200 | Accuracy | 46.0% |

##### General Benchmarks (for regression testing)
| Dataset | MedGemma 4B | MedGemma 27B |
|---------|-------------|--------------|
| MMLU Pro | 39.1% | 60.2% |
| Global MMLU Lite | 55.5% | 74.5% |
| MMMU validation | 47.3% | — |

##### MedSigLIP Zero-Shot AUC (standalone encoder, not MedGemma 4B as a whole)
Average AUC across 13 CXR findings: **0.844**
| Finding | AUC |
|---------|-----|
| Enlarged Cardiomediastinum | 0.858 |
| Cardiomegaly | 0.904 |
| Lung Opacity | 0.931 |
| Consolidation | 0.880 |
| Edema | 0.891 |
| Pneumothorax | 0.862 |
| Pleural Effusion | 0.914 |
| Fracture | 0.708 |

**Emerging Patterns**:
- Google uses **tokenized F1** (not accuracy) as primary VQA metric — this is more robust for open-ended answers
- RadGraph F1 is the standard for report generation, not ROUGE
- PathVQA is conspicuously ABSENT from MedGemma's evaluation
- The model uses a specialized MedSigLIP vision encoder (SigLIP fine-tuned on medical data)

#### Round 3: Crystallization

**Questions Asked**:
- How does MedSigLIP compare to the standard SigLIP encoder?
- What training data was used (public vs. proprietary)?
- What is fine-tuning headroom?

**Final Understanding**:

MedGemma 4B = Gemma 3 4B LLM + MedSigLIP vision encoder (SigLIP fine-tuned on medical images). The vision encoder was pretrained on diverse medical images including chest X-rays, dermatology, ophthalmology, histopathology, and CT/MRI slices.

Training data (public portion):
- MIMIC-CXR (chest X-rays + reports)
- SLAKE-VQA (radiology VQA training set)
- PAD-UFES-20 (skin lesions)
- SCIN (dermatology, Google Health + Stanford)
- TCGA (cancer genomics)
- CAMELYON (histopathology)
- PMC-OA (biomedical literature figures)
- Mendeley Digital Knee X-Ray
- MedQA, VQA-RAD, MedXpertQA, AfriMed-QA (text QA)

**Fine-tuning results** (Table 13):
| Task | OOB RadGraph F1 | Fine-tuned RadGraph F1 |
|------|-----------------|----------------------|
| MIMIC-CXR Report Generation | 29.5% | 30.3% |

| Task | OOB Accuracy | Fine-tuned Accuracy |
|------|--------------|---------------------|
| SIIM-ARC Pneumothorax | 85.9% | 87.8% |

| Task | OOB F1 | Fine-tuned F1 |
|------|--------|---------------|
| CRC100k Histopathology | 32.8% | 94.5% |

The CRC100k histopathology case is dramatic — OOB 32.8% → fine-tuned 94.5% — demonstrating substantial latent capability.

**Validated Assumptions**:
- MedGemma 4B pretrained (PT) variant is better for generation tasks; instruction-tuned (IT) is better for VQA
- SLAKE's closed-ended accuracy (87.6%) is dramatically higher than open-ended recall (63.3%), which is typical
- VQA-RAD's tokenized F1 (49.9%) looks low but is the community standard — closed accuracy (69.1%) is more comparable to older papers

---

### Opening Item 2: Public Medical VQA Datasets

#### Round 1: Surface Exploration

**Questions Asked**:
- What VQA datasets cover medical imaging with public access?
- How are LLaVA-Med and similar models evaluated on VQA-RAD/PathVQA/SLAKE?
- What are the dataset sizes, modalities, and question types?

**Key Discoveries**:

**VQA-RAD**
- **Source**: National Library of Medicine, manually annotated
- **Size**: 315 images, 3,515 QA pairs
- **Modalities**: Radiology (CT, MRI, chest X-ray, head CT)
- **Question types**: 637 open-ended, 878 closed-ended (primarily yes/no)
- **Images per patient**: ~11 on average
- **HuggingFace**: `flaviagiammarino/vqa-rad`
- **Split note**: Original splits have train/test image contamination (same patients appear in both) — use Yang et al. (2024) splits for rigorous evaluation
- **Primary metric**: Tokenized F1 for overall; accuracy for closed-ended

**SLAKE (Semantically-Labeled Knowledge-Enhanced)**
- **Source**: ISBI 2021 paper; bilingual English-Chinese
- **Size**: 642 radiology images (282 CT, 181 MRI, 179 X-ray), 14,028 QA pairs
- **Modalities**: CT, MRI, X-ray across 5 body regions (head, neck, chest, abdomen, pelvis)
- **Question types**: Closed-ended (limited answer choices) and open-ended (free text)
- **HuggingFace**: `Voxel51/SLAKE`, `mdwiratathya/SLAKE-vqa-english`
- **Official site**: med-vqa.com/slake
- **Primary metric**: Tokenized F1; closed-ended accuracy separately

**PathVQA**
- **Source**: MICCAI 2020 paper
- **Size**: 4,998 pathology images, 32,799 QA pairs
- **Modalities**: Histopathology (H&E stained tissue)
- **Question types**: Yes/no (closed) and open-ended (anatomical structures, diseases)
- **HuggingFace**: `flaviagiammarino/path-vqa`
- **Leaderboard metrics**: Yes/No Accuracy, Free-form accuracy, Overall accuracy
- **Note**: NOT evaluated in MedGemma's official report despite being widely used in other VLM evals

**OmniMedVQA (CVPR 2024)**
- **Source**: 73 medical classification datasets converted to QA via GPT-3.5
- **Size**: 118,010 images, 127,995 QA items
- **Modalities**: 12 different medical imaging modalities
- **Body coverage**: 20+ anatomical regions
- **Question types**: Modality Recognition, Anatomy Identification, Disease Diagnosis, Lesion Grading, Other Biological Attributes
- **HuggingFace**: `foreverbeliever/OmniMedVQA`
- **Key finding**: Medical-specialized LVLMs perform *worse* than general-domain models, suggesting need for better training

**MedXpertQA**
- **Source**: Google internal benchmark
- **Size**: 2,450 examples (2,000 with images)
- **Purpose**: Out-of-distribution hard evaluation set
- **Metric**: Accuracy on multiple-choice
- **Key feature**: Designed to be hard even for strong models (MedGemma 4B gets 24.4% multimodal, 14.2% text-only)

**Initial Gaps**: Need to understand evaluation harness implementations and which datasets are freely usable vs. require data access agreements.

#### Round 2: Deep Dive

**Questions Asked**:
- What open-ended vs. closed-ended evaluation approaches exist?
- How does tokenized F1 differ from exact match?
- What are the data access requirements?

**Key Discoveries**:

**Access Requirements**:
- VQA-RAD: Publicly downloadable, no registration required
- SLAKE: Publicly downloadable at med-vqa.com/slake
- PathVQA: Publicly accessible on GitHub/HuggingFace
- OmniMedVQA: Available at `github.com/OpenGVLab/Multi-Modality-Arena`
- MIMIC-CXR: Requires PhysioNet credentialed access (HIPAA-compliant research)
- PathMCQA, US-Derm MCQA: Google internal — not publicly available
- EyePACS: Public (kaggle.com/c/diabetic-retinopathy-detection)

**Evaluation Approaches**:

Closed-ended (binary yes/no): Exact match after normalization
Open-ended: Tokenized F1 (like token-level F1 in reading comprehension) — allows partial credit, handles synonyms better than exact match

Tokenized F1 formula: `F1 = 2 * (precision * recall) / (precision + recall)` where precision/recall are computed over shared tokens between prediction and reference.

**Emerging Patterns**:
- SLAKE and VQA-RAD are the most used duo in recent papers (2023-2025)
- PathVQA is considered a harder benchmark due to pathology image complexity
- The trend is moving from accuracy → tokenized F1 for open-ended, while keeping accuracy for closed-ended

#### Round 3: Crystallization

**Questions Asked**: How do SLAKE's English-only vs. bilingual versions affect evaluation?

**Final Understanding**:

For any new evaluation of MedGemma 4B against these benchmarks:

1. **SLAKE**: Use English-only split (`SLAKE-vqa-english`). Report both Overall Token F1 and Closed-ended Accuracy separately. The official MedGemma test set is SLAKE test split (1,061 images).

2. **VQA-RAD**: Use Yang et al. (2024) splits, not original splits. Report tokenized F1 for overall and accuracy for closed Q&A subset. MedGemma uses 2,248 test images.

3. **PathVQA**: Freely available, not used by Google but good for pathology evaluation. Report Yes/No Accuracy, Free-form Accuracy, Overall Accuracy.

4. **OmniMedVQA**: Most comprehensive benchmark (127K QA pairs, 12 modalities) — good for stress testing generalization. Available on HuggingFace.

---

### Opening Item 3: Standard Metrics for Medical Imaging AI Beyond ROUGE-L

#### Round 1: Surface Exploration

**Questions Asked**:
- What metrics does MONAI support for medical image analysis?
- What clinical NLP metrics exist in HuggingFace evaluate?
- What general-purpose evaluation harnesses cover medical tasks?

**Key Discoveries**:

MONAI (medical imaging framework) supports comprehensive metrics:
- **Segmentation**: Dice Coefficient, IoU, Hausdorff Distance (HD95), Surface Dice, Average Surface Distance
- **Classification**: AUC-ROC, Average Precision, Confusion Matrix → Sensitivity/Specificity, Matthews Correlation Coefficient, Balanced Accuracy, Cohen's Kappa
- **Regression**: MSE, MAE, RMSE, PSNR, SSIM

HuggingFace `evaluate` library: ROUGE, BLEU, BERTScore, METEOR — all general-purpose, no medical-specific metrics natively.

lm-evaluation-harness: Has `mimic_repsum` (ROUGE + BERTScore for report summarization), `medmcqa`, `medqa`, `pubmedqa` — all text-only. No medical VQA tasks.

**Initial Gaps**: Need the clinical-specific metrics (RadGraph, CheXbert, GREEN score) that go beyond standard NLP metrics.

#### Round 2: Deep Dive

**Questions Asked**:
- What is RadGraph F1 and how is it computed?
- What is the GREEN score?
- What other clinical metrics exist for report generation?
- How is CheXbert used for evaluation?

**Key Discoveries**:

**RadGraph F1** (Jain et al., 2021):
- Converts free-text radiology reports into structured entity-relation graphs using NER + RE models
- Computes precision/recall over shared (entity, relation) pairs between generated and reference report
- F1 = harmonic mean of entity-level precision and recall
- Strongest correlation with radiologist evaluation (Kendall tau 0.515-0.531)
- Limitation: Only trained on chest X-ray domain; does not generalize to other modalities
- MedGemma 4B achieves 29.5% RadGraph F1 on MIMIC-CXR (near SOTA; MedVersa SOTA is 30.0%)
- Available as pip package: can be integrated into evaluation pipelines

**GREEN Score** (Yu et al., 2024, EMNLP Findings):
- Uses LLM (fine-tuned on radiology domain) to identify and explain clinically significant errors
- Provides both quantitative score AND human-readable error explanations
- Training: 100,000 reference-generated report pairs from 6 chest X-ray datasets
- Datasets used in training: MIMIC-CXR, MIMIC-PRO, CandidPTX, PadChest, BIMCV-covid19, OpenI
- Performance: Matches GPT-4 in radiologist error count correlation, outperforms RadGraph F1 in expert preference alignment
- Available as `pip install green-score` (open source, lightweight)
- Limitation: Like RadGraph, currently chest X-ray focused

**RaTEScore** (2024, EMNLP):
- Entity-aware metric emphasizing crucial medical entities (diagnostic outcomes, anatomical details)
- Addresses RadGraph F1's weakness with synonyms
- Newer approach but less adopted in current evaluation pipelines

**F1-CheXpert (CheXpert Labeler)**:
- Uses NLP label extraction model trained on CheXpert's 14 pathology labels
- Labels: Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding
- Computes label-level F1 between predicted and reference report labels
- Limitation: Limited to 14 CheXpert labels; binary/ternary labels lose nuance
- Less preferred than RadGraph F1 in current literature

**BERTScore**:
- Uses contextual BERT embeddings to compute similarity
- Correlates reasonably with human judgment (second after RadGraph)
- Domain-general, works across modalities
- Available in HuggingFace evaluate: `evaluate.load('bertscore')`

**Clinical Classification Metrics** (for image classification tasks):
- **Macro-F1**: Primary metric for multi-class classification (MedGemma uses for CXR 5-condition task)
- **AUC-ROC**: Primary for binary classification (MedSigLIP uses for zero-shot CXR finding detection)
- **Cohen's Kappa**: Agreement metric; used in this project's PRESCREEN/VALIDATE evaluation
- **Balanced Accuracy**: More robust than accuracy for imbalanced datasets

**Emerging Patterns**:
- For VQA: Tokenized F1 > Accuracy (better handles open-ended answers)
- For report generation: RadGraph F1 > ROUGE-L (clinical relevance)
- For classification: Macro-F1 and AUC are standard; accuracy is secondary
- GREEN score is becoming the recommended standard for report generation but needs more tooling adoption

#### Round 3: Crystallization

**Final Metric Recommendations by Task**:

| Task Type | Primary Metric | Secondary Metrics | Notes |
|-----------|---------------|-------------------|-------|
| Medical VQA (closed) | Accuracy | — | Binary yes/no questions |
| Medical VQA (open) | Tokenized F1 | BLEU-4 | Per Google's MedGemma approach |
| Radiology report generation | RadGraph F1 | GREEN score, BERTScore | RadGraph F1 is current standard |
| CXR classification | Macro-F1 | AUC-ROC | For 5-14 conditions |
| Medical image classification | Accuracy | Balanced Accuracy | MCQ format |
| Segmentation | Dice Coefficient | HD95, Surface Dice | From MONAI |
| Retrieval | AUC-ROC | Mean Average Precision | 1-vs-all for multiclass |

**ROUGE-L is NOT recommended** as a primary metric for medical tasks. It has low clinical relevance and poor correlation with expert judgment. It may be reported as a secondary metric for backwards compatibility with older literature.

---

### Opening Item 4: Radiology Report Generation Benchmarks

#### Round 1: Surface Exploration

**Questions Asked**:
- How is MIMIC-CXR structured for report generation?
- What is the evaluation protocol for radiology report generation?
- What benchmarks use RadGraph or clinical metrics?

**Key Discoveries**:

**MIMIC-CXR Dataset Structure** (via MIT-LCP/mimic-code):
- Contains chest X-ray DICOM images + de-identified radiology reports
- 377,110 DICOMs, 227,835 studies, 65,379 subjects
- Standard split file: `mimic-cxr-2.0.0-split.csv.gz`
- Reports have FINDINGS and IMPRESSION sections (model must generate both)
- Time period: 2012-2016, Beth Israel Deaconess Medical Center
- Access: PhysioNet credentialed access required (HIPAA-compliant)

**MedGemma Evaluation Protocol for Report Generation**:
- 912 cases used for RadGraph F1 evaluation
- 306 cases for human expert evaluation
- Uses pretrained model (not instruction-tuned) to match MIMIC-CXR reporting style
- Human evaluation: radiologists compare generated vs. original for clinical decision equivalence

**Initial Gaps**: What alternative datasets to MIMIC-CXR exist that don't require credentialed access?

#### Round 2: Deep Dive

**Questions Asked**:
- What open-access radiology report datasets exist?
- How does the GREEN score work in practice?
- What is the CheXbert labeler evaluation pipeline?

**Key Discoveries**:

**GREEN Score** was trained on 6 datasets (100K pairs):
1. MIMIC-CXR (primary)
2. MIMIC-PRO (extended)
3. CandidPTX (pneumothorax focused)
4. PadChest (Spanish, multilingual)
5. BIMCV-covid19 (COVID chest X-ray)
6. OpenI (Indiana University CXR — OPEN ACCESS, no credentialing required)

**OpenI (Indiana University CXR)**: Important alternative to MIMIC-CXR as it is publicly available without PhysioNet credentials. Contains ~7,470 chest X-ray images with reports.

**CheXpert Dataset for Classification** (Stanford, 2019):
- 224,316 chest radiographs from 65,240 patients
- 14 observation labels (positive/negative/uncertain)
- Used for zero-shot and few-shot classification evaluation
- Access: Requires Stanford research agreement (free, online form)

**ROCO Dataset** (used in Visual Med-Alpaca):
- 65,444 radiology image-caption pairs for training
- 8,179 for validation
- Open access; used for image captioning fine-tuning

**Standard Evaluation Pipeline for Radiology Report Generation**:
1. Generate FINDINGS + IMPRESSION text from chest X-ray image
2. Compute RadGraph F1 (primary) using radgraph Python package
3. Optionally compute: ROUGE-L, BLEU-4, BERTScore (for secondary comparison)
4. Optionally compute: GREEN score for interpretable error analysis
5. Human expert evaluation for clinical utility (ground-truth validation)

#### Round 3: Crystallization

**Questions Asked**: What specialized report generation benchmarks exist beyond MIMIC-CXR?

**Final Understanding**:

**Radiology Report Generation Benchmarks Ranked by Accessibility**:

1. **MIMIC-CXR** (gold standard, requires PhysioNet access)
   - 227K studies; standard train/val/test splits defined in split CSV
   - Metric: RadGraph F1 (primary), ROUGE-L (secondary)
   - Used by: MedGemma, MAIRA, CheXagent, MedVersa, R2Gen, etc.

2. **OpenI / Indiana University CXR** (open access, no registration)
   - ~7,470 images; smaller but unrestricted
   - Reports structured similarly to MIMIC-CXR
   - Good for quick prototyping without credentialing overhead

3. **ROCO** (open access)
   - 65K radiology images from PMC with figure captions
   - Not radiology reports per se — figure captions are shorter and less clinical
   - Good for image captioning; limited for clinical report generation evaluation

4. **PadChest** (open access, Spanish)
   - 160K chest X-rays with Spanish radiologist reports
   - 174 findings, 19 differential diagnoses, 104 anatomic locations
   - Good for multilingual evaluation

**Key Insight on MedGemma Report Generation**:
The pretrained (PT) variant is used for report generation because:
- RadGraph F1 is sensitive to reporting style
- MIMIC-CXR training data uses a specific formatting convention
- Instruction-tuning shifts the model toward Gemma-style conversation format
- PT model preserves the training distribution alignment needed for RadGraph F1 scoring

Fine-tuning on MIMIC-CXR-specific LoRA increases RadGraph F1 from 29.5% → 30.3%, a meaningful improvement in this tight SOTA range.

---

## Cross-Cutting Insights

### 1. The Two-Metric System: Open-Ended vs. Closed-Ended

All medical VQA datasets (VQA-RAD, SLAKE, PathVQA) share a fundamental split between closed-ended (yes/no, multiple choice) and open-ended (free-text) questions. These require different metrics and are typically reported separately:
- **Closed-ended**: Exact match accuracy (simple, reproducible, high scores expected)
- **Open-ended**: Tokenized F1 (partial credit, handles synonyms, lower scores expected)

Never report a single "overall accuracy" without this breakdown — it conflates very different evaluation regimes.

### 2. Clinical Metrics Outperform Textual Overlap Metrics

Across all research rounds, there is consistent evidence that:
- ROUGE-L / BLEU-4 are poorly aligned with clinical judgment for report generation
- RadGraph F1 (entity-relation graph overlap) has 2x higher correlation with expert preference
- GREEN score (LLM-based error annotation) adds interpretability
- For classification, Macro-F1 and AUC-ROC are far more clinically meaningful than accuracy

This is especially important for report generation tasks — **do not use ROUGE-L as the primary metric**.

### 3. Dataset Contamination Risk

VQA-RAD's original train/test split has image contamination (same patients appear in both splits). MedGemma explicitly uses Yang et al. (2024) cleaned splits. Any evaluation on VQA-RAD should cite which split was used.

SLAKE's training data overlaps with the evaluation domain — models trained on SLAKE training set may have inflated test set performance.

### 4. MMMU as a General Multimodal Benchmark

MMMU (HuggingFace: `MMMU/MMMU_Pro`) provides Health and Medicine subcategories:
- Basic Medical Science
- Clinical Medicine
- Diagnostics and Laboratory Medicine
- Pharmacy
- Public Health

These are multiple-choice with images, evaluated by accuracy. MMMU-Pro is harder (10 options vs. 4, vision-only mode). MedGemma 4B achieves 47.3% on MMMU validation. This is a good orthogonal benchmark to VQA-RAD/SLAKE since it covers medical knowledge beyond pure radiology.

### 5. OmniMedVQA: Most Comprehensive Freely Available Benchmark

CVPR 2024's OmniMedVQA (127K QA pairs, 12 modalities, 20 body regions) is the most comprehensive public benchmark available without credentialing. Key finding from paper: medical-specialized VLMs often perform *worse* than general models, suggesting architectural/training issues rather than just data issues.

---

## Architecture/Design Decisions

### Decision 1: MedGemma Uses Tokenized F1, Not Exact Match or ROUGE

**Rationale**: Medical terminology has many equivalent expressions. Tokenized F1 gives partial credit and is more robust than exact match for open-ended clinical answers.

**Trade-off**: Tokenized F1 is implementation-specific (tokenizer choices matter). ROUGE-L is more widely comparable across papers but less clinically valid.

**Implication for this project**: If benchmarking MedGemma 4B on VQA tasks, implement tokenized F1, not simple accuracy. This explains why VQA-RAD F1 (49.9%) looks low — it's a harder metric than closed-set accuracy (69.1%).

### Decision 2: Pretrained Model for Report Generation, IT for VQA

**Rationale**: Instruction tuning shifts output style toward conversational format, which hurts RadGraph F1 that expects clinical report formatting.

**Trade-off**: You cannot use a single model variant for all tasks; need to select the right variant for each task type.

### Decision 3: RadGraph F1 over ROUGE-L for Report Generation

**Rationale**: RadGraph F1 captures clinical entity correctness (disease mentions, anatomy references, severity) rather than surface text overlap. Correlation with expert judgment is significantly higher.

**Trade-off**: RadGraph is computationally heavier (entity-relation extraction model required) and chest X-ray specific.

---

## Edge Cases and Limitations

1. **RadGraph chest X-ray specificity**: RadGraph F1 was trained only on chest X-ray reports. It cannot evaluate MRI, CT, or pathology reports reliably. For other modalities, fall back to ROUGE-L/BERTScore or use GREEN score.

2. **SLAKE test set contamination risk**: If your model trained on SLAKE training set, the evaluation numbers will be inflated vs. truly out-of-distribution patients. Use SLAKE only as an in-distribution benchmark, not as OOD evaluation.

3. **VQA-RAD split selection is critical**: Using original vs. Yang et al. (2024) splits gives significantly different numbers. Always specify which split was used.

4. **PathVQA is NOT in MedGemma's official evaluation**: Using PathVQA provides an independent benchmark that is less subject to training data leakage, but no official MedGemma baseline to compare against.

5. **MIMIC-CXR requires credentialing**: Cannot use MIMIC-CXR programmatically without PhysioNet approval. For open development/testing without credentials, use OpenI instead.

6. **MedXpertQA is proprietary**: Google's hardest OOD benchmark is not publicly released, making it impossible to reproduce their MedXpertQA numbers.

7. **Tokenized F1 is implementation-sensitive**: The exact tokenizer used (whitespace split, clinical NLP tokenizer, etc.) affects F1 scores. Should use the same tokenizer as the original paper to produce comparable results.

---

## Recommendations for This Project

### For Immediate Benchmarking of MedGemma 4B Multimodal

**Most practical benchmark stack** (all publicly accessible, no credentials):

| Priority | Dataset | HuggingFace Path | Metric | Why |
|----------|---------|-----------------|--------|-----|
| P0 | SLAKE English | `Voxel51/SLAKE` | Tokenized F1 + Closed Accuracy | Google's primary VQA benchmark |
| P0 | VQA-RAD (Yang splits) | `flaviagiammarino/vqa-rad` | Tokenized F1 + Closed Accuracy | Google's secondary VQA benchmark |
| P1 | MMMU Health & Medicine | `MMMU/MMMU_Pro` | Accuracy | Broad medical knowledge, easy to run |
| P1 | PathVQA | `flaviagiammarino/path-vqa` | Yes/No Accuracy + Open F1 | Pathology-specific, independent of Google's training |
| P2 | OmniMedVQA | `foreverbeliever/OmniMedVQA` | Accuracy | Most comprehensive, 12 modalities |
| P3 | OpenI CXR | Direct download | RadGraph F1 | Report generation, no credentials needed |

### For Report Generation Evaluation

Use RadGraph F1 as primary metric. Install via: `pip install radgraph`.
GREEN score is strongly recommended as secondary for interpretable error analysis: `pip install green-score`.

Do NOT use ROUGE-L as a primary metric for clinical report quality. It can be reported as a historical comparison point only.

### For This Project's Clinical Trial Matching Use Case

The MedGemma 4B multimodal capabilities most relevant to this project:
- **CXR classification** (Macro-F1 88.9% on MIMIC-CXR): Can be used to extract imaging findings from patient X-rays for INGEST
- **Medical VQA** (SLAKE F1 72.3%): Can answer structured questions about medical images for structured profile extraction
- **Text QA** (MedQA 64.4%): The 4B model is less capable for pure text reasoning — use 27B for clinical trial criterion matching

---

## Open Questions

1. **Yang et al. (2024) VQA-RAD splits**: The exact citation and dataset card for these cleaned splits is not publicly documented in a way that makes them easy to reproduce. Need to find the specific HuggingFace dataset version or paper for the corrected splits.

2. **GREEN score generalization**: GREEN was trained primarily on chest X-rays. Its applicability to other radiology modalities (CT, MRI) and pathology reports is unvalidated.

3. **PathVQA current SOTA**: No official MedGemma baseline exists for PathVQA. The literature SOTA as of 2024 is in the 88-92% range for yes/no questions with specialized models, but this project lacks a verified comparison point.

4. **MedXpertQA availability**: Google has not released MedXpertQA publicly, limiting its use as a benchmark for external comparison.

---

## Research Methodology Notes

- **Total Deepwiki rounds**: 5 (20 queries)
- **Web searches**: 6 targeted queries
- **Web fetches**: 2 (MedGemma technical report arXiv page, HuggingFace model card)
- **Primary challenge**: The official MedGemma repository (google-health/medgemma) is not indexed by DeepWiki, requiring web search fallback for the most critical source
- **Repositories that were indexed**: microsoft/LLaVA-Med, MMMU-Benchmark/MMMU, Project-MONAI/MONAI, EleutherAI/lm-evaluation-harness, mahmoodlab/CONCH, mahmoodlab/UNI, MIT-LCP/mimic-code, google-research/google-research, huggingface/evaluate, google-deepmind/gemma, cambridgeltl/visual-med-alpaca
- **Quality confidence level**: HIGH for MedGemma official benchmarks (sourced directly from arXiv technical report); HIGH for dataset characteristics (cross-referenced multiple sources); MEDIUM for metric implementations (theoretical understanding solid, implementation specifics may vary); HIGH for dataset access paths (HuggingFace paths verified via web search)

---

## Sources

- [MedGemma Technical Report (arXiv:2507.05201)](https://arxiv.org/html/2507.05201v2)
- [MedGemma 4B PT Model Card (HuggingFace)](https://huggingface.co/google/medgemma-4b-pt)
- [MedGemma 1.5 Model Card (Google Developers)](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [GREEN Score Paper (EMNLP 2024 Findings)](https://arxiv.org/abs/2405.03595)
- [GREEN Score Homepage (Stanford AIMI)](https://stanford-aimi.github.io/green.html)
- [RaTEScore Paper (EMNLP 2024)](https://www.medrxiv.org/content/10.1101/2024.06.24.24309405v2.full)
- [OmniMedVQA Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_OmniMedVQA_A_New_Large-Scale_Comprehensive_Evaluation_Benchmark_for_Medical_LVLM_CVPR_2024_paper.pdf)
- [PathVQA HuggingFace Dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
- [VQA-RAD HuggingFace Dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [SLAKE HuggingFace Dataset](https://huggingface.co/datasets/Voxel51/SLAKE)
- [SLAKE Official Site](https://www.med-vqa.com/slake/)
- [SLAKE Paper (arXiv)](https://arxiv.org/abs/2102.09542)
- [OmniMedVQA HuggingFace Dataset](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA)
- [Radiology Report Evaluation Progress Review (PubMed)](https://pubmed.ncbi.nlm.nih.gov/37720336/)
- [Medical Multimodal Evaluation Data (HuggingFace)](https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data)
- [AdaptLLM BioMed VQA Benchmark (HuggingFace)](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)
- [DeepWiki: microsoft/LLaVA-Med](https://deepwiki.com/wiki/microsoft/LLaVA-Med#1)
- [DeepWiki: MMMU-Benchmark/MMMU](https://deepwiki.com/wiki/MMMU-Benchmark/MMMU#2)
- [DeepWiki: EleutherAI/lm-evaluation-harness](https://deepwiki.com/wiki/EleutherAI/lm-evaluation-harness#4)
- [DeepWiki: mahmoodlab/CONCH](https://deepwiki.com/wiki/mahmoodlab/CONCH#5)
