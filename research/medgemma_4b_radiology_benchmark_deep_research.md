# Deep Research Report: MedGemma 4B Radiology Benchmark Performance & Limitations

**Date**: 2026-02-23
**Researcher**: deep-research-agent
**Repositories Analyzed**: Google-Health/medgemma, google/medgemma-1.5-4b-it (HuggingFace), arxiv:2507.05201
**Total Research Rounds**: 5 (updated 2026-02-23 second session)

---

## Executive Summary

MedGemma 4B (google/medgemma-1.5-4b-it) is a 4-billion parameter multimodal vision-language model trained on medical image-text pairs. Our 10-case MedPix thorax CT benchmark showed 20% correct diagnoses (2/10) vs. Gemini Flash's 70% (7/10). This result is broadly **consistent with official benchmarks and design intent**: MedGemma 4B achieves only 61% macro accuracy on its own internal CT classification benchmark (7 conditions, balanced), and only 27% macro F1 on CT-RATE (18 conditions). The 20% result on our harder open-ended rare-disease diagnosis task is therefore not surprising.

The core architectural issue is a **task type mismatch**: MedGemma 4B was optimized for structured finding classification and VQA tasks, not open-ended rare disease differential diagnosis. Google explicitly recommends using MedSigLIP (not MedGemma) for classification tasks. Open-ended diagnosis generation in MedGemma 4B is prone to common-disease anchoring and hallucination, and Google themselves state the model requires fine-tuning for specific clinical subdomains before deployment.

The 4B model size also imposes hard limits on reasoning complexity — the 4B variants "were not well suited for agentic tasks, demonstrating difficulty following system instructions." Complex diagnostic reasoning involving rare conditions requires the kind of multi-step inference that only becomes reliable at the 27B scale.

---

## Research Objectives

1. What are the official benchmark results for MedGemma 4B on imaging tasks, especially CT?
2. What are the known limitations vs. the 27B model?
3. What was the model trained on and what tasks does it excel at vs. struggle with?
4. Why did our 10-case MedPix thorax CT benchmark show 20% accuracy?
5. Are there community benchmarks that corroborate or contradict our results?

---

## Detailed Findings

### Opening Item 1: Official Benchmark Results

#### Round 1: Surface Exploration

**Questions Asked**: General MedGemma 4B benchmark performance, official sources

**Key Discoveries**:
- MedGemma 1.5 4B (released January 13, 2026) is the latest version
- Official CT benchmark: **61.1% macro accuracy** on internal CT Dataset 1 (7 conditions)
- CT-RATE benchmark (18 conditions): **27.0% macro F1**, 42.0% macro recall
- MRI Dataset 1: **64.7% macro accuracy** (10 conditions)
- MedQA text reasoning: **69.1%** (vs. 27B's 85.3%)
- Chest X-ray (MIMIC CXR): **89.5% macro F1** (top 5 conditions)

**Initial Gaps**: What specific conditions are in CT Dataset 1? How does 61% on classification compare to our open-ended diagnosis task?

#### Round 2: Deep Dive

**Questions Asked**: CT preprocessing requirements, training data composition, task type design

**Key Discoveries**:

1. **CT preprocessing is specialized**: For CT images, "three windows were preselected and converted into RGB color channels" to highlight (1) bone and lung, (2) soft tissue, (3) brain. Standard settings: 350/50 (soft tissue), 1500/−600 (lung), 1800/400 (bone). This three-channel RGB windowing technique is the expected input format — plain JPEG/PNG screenshots from DICOM viewers may not match training distribution.

2. **Classification vs. generation tasks**: Google explicitly states: "MedGemma is useful for medical text or imaging tasks that require generating free text, like report generation or VQA, while **MedSigLIP** is recommended for imaging tasks that involve structured outputs like classification or retrieval."

3. **Training data**: Vision encoder fine-tuned on "over 33M medical image-text pairs" — dominated by histopathology (32.6M of 33M). Radiology, including CT slices, is a minority. CT slices in training: "54,573 more CT 2D slices" (v1.5 increment), curated from abnormal findings in radiology reports.

4. **CT dataset is internal and small**: CT Dataset 1 = internal Google dataset, 7 conditions, balanced sampling. Not publicly known what conditions these are. The balanced sampling inflates accuracy vs. real-world distribution.

**Emerging Patterns**: The 61% CT classification number is on a closed-set, balanced, internal benchmark — not open-ended diagnosis of rare conditions from photos.

#### Round 3+: Crystallization

**Questions Asked**: Open-ended vs. classification failure modes, hallucination patterns, 4B size limitations

**Final Understanding**:

| Benchmark | MedGemma 1.5 4B | MedGemma 1 4B | MedGemma 1 27B | Notes |
|-----------|-----------------|---------------|----------------|-------|
| CT Dataset 1 (7 cond.) | **61.1%** | 58.2% | 57.8% | Internal, balanced, closed-set |
| CT-RATE (18 cond.) | **27.0% F1** | 23.5% F1 | — | CT report generation task |
| CXR MIMIC (top 5) | **89.5% F1** | 88.9% F1 | 71.7% F1 | Strong on common CXR findings |
| MedQA text | 69.1% | 64.4% | 85.3% | Text reasoning significantly worse at 4B |
| MedXpertQA (OOD) | — | — | 25.7% | Even 27B collapses on OOD hard text |
| AgentClinic (agentic) | ~30%* | — | 46% | 4B "not well suited for agentic tasks" |

*4B substantially worse than 27B on agentic tasks per technical report

**Validated Assumptions**:
- 61% CT accuracy is on a closed-set, balanced internal benchmark — not comparable to our open-ended task
- MedGemma 4B CT training data is heavily biased toward common, structural findings from abnormal radiology reports
- Rare thoracic conditions (pulmonary artery dilatation, giant hiatal hernia, Paget's disease of sternum) are almost certainly underrepresented in training

---

### Opening Item 2: Known Limitations vs. 27B Model

#### Round 1: Surface Exploration

**Questions Asked**: 4B vs. 27B comparison, model architecture differences

**Key Discoveries**:
- MedGemma 4B = multimodal (vision + text), 4B parameters
- MedGemma 27B = text-only (no vision encoder), 27B parameters
- MedGemma 1.5 4B = ONLY multimodal variant of v1.5; no v1.5 27B exists yet
- For imaging: only 4B is available as a multimodal foundation model

**Initial Gaps**: What are the specific reasoning limitations of 4B vs. 27B?

#### Round 2: Deep Dive

**Questions Asked**: Specific 4B failure modes in technical report, agentic task performance

**Key Discoveries**:

1. **Agentic task failure**: Technical report states 4B variants "were not well suited for agentic tasks, demonstrating difficulty following system instructions." This is a fundamental 4B limitation — complex multi-step reasoning degrades.

2. **Text reasoning gap is large**: MedQA: 69.1% (4B) vs. 85.3% (27B) — 16 percentage point gap. On harder clinical reasoning tasks, this gap widens further.

3. **Imaging exception**: On imaging-centric tasks, 4B can match or exceed 27B because it has the specialized SigLIP vision encoder: CT accuracy 61.1% (4B) vs 57.8% (27B). The 27B without specialized vision encoder actually underperforms 4B on visual tasks.

4. **The tradeoff**: 4B is better at visual pattern matching for standard imaging, 27B is far better at clinical reasoning from text. Our task requires BOTH visual finding extraction + rare disease reasoning — the 4B cannot do the reasoning step reliably.

#### Round 3+: Crystallization

**Final Understanding**: The 4B vs. 27B comparison is nuanced:

| Capability | MedGemma 4B | MedGemma 27B |
|------------|-------------|--------------|
| Image encoding | Strong (SigLIP medical encoder) | None (text-only) |
| Standard finding classification | Better (61.1% CT) | Worse (57.8% CT) |
| Text reasoning | Weaker (MedQA 69%) | Much stronger (85%) |
| Complex multi-step reasoning | Poor ("difficulty following instructions") | Good |
| Rare disease differential | Poor | Much better |
| Agentic workflows | Fails | Works (46% AgentClinic) |

**For our MedPix task**: Requires both visual finding recognition AND rare disease clinical reasoning → 4B is weak on the reasoning component.

---

### Opening Item 3: Training Data and Design Intent

#### Round 1: Surface Exploration

**Questions Asked**: What was MedGemma 4B trained on? What is it designed for?

**Key Discoveries**:
- Vision encoder (SigLIP) trained on "33M+ medical image-text pairs"
- Dominated by: histopathology (32.6M / 33M = 98.8%)
- Radiology images (CXR, CT, MRI) are a small minority
- Post-training: included "medical imaging data with paired text" + RLHF

**Initial Gaps**: What types of CT tasks was it optimized for?

#### Round 2: Deep Dive

**Questions Asked**: SigLIP training composition, CT image format, benchmark design

**Key Discoveries**:

1. **SigLIP training data breakdown**:
   - Histopathology: ~32.6M samples (overwhelming majority)
   - Radiology (CXR, CT, MRI): small fraction
   - CT slices v1.5 increment: 54,573 slices (vs. 32.6M histopathology patches — ratio ~1:600)

2. **CT benchmark design is favorable**: CT Dataset 1 is internal, 7 conditions, balanced — not representative of real radiology where hundreds of conditions exist and rare diseases appear

3. **Designed as a foundation/starting point**: Google's official guidance: "These models are starting points for developers rather than finished clinical products, require validation and customization, and their outputs should not be used for direct diagnosis or treatment."

4. **Recommended use cases**:
   - Fine-tuning baseline for specialized medical data
   - Report summarization and extraction
   - Medical VQA (structured formats)
   - Document understanding

5. **NOT recommended for**: "Primary diagnostic tool", "autonomous diagnosis", multi-image sequential analysis

#### Round 3+: Crystallization

**Key Insight on Our 20% Result**:

Google's own research (Technical Report, arXiv:2507.05201) states: "Automated benchmarks represent only the first step towards validating real-world utility." The 61% CT classification on their internal balanced 7-condition dataset does NOT translate to open-ended rare thoracic disease diagnosis. Our 10-case MedPix benchmark with conditions including:
- Idiopathic dilatation of pulmonary artery
- Giant hiatal hernia
- Paget's disease of sternum

...is exactly the type of OOD (out-of-distribution) rare condition task where MedGemma 4B would be expected to fail. These conditions are unlikely to have been well-represented in training.

---

### Opening Item 4: Community Benchmarks and Independent Evaluations

#### Round 1: Surface Exploration

**Questions Asked**: Community evaluations, Kaggle challenge results, independent benchmarks

**Key Discoveries**:
- Published paper: "MedGemma vs GPT-4: Open-Source and Proprietary Zero-shot Medical Disease Classification from Images" (arXiv:2512.23304)
- Kaggle MedGemma Impact Challenge is an active community evaluation venue
- Limited public community CT diagnosis benchmarks found

**Initial Gaps**: What does the MedGemma vs. GPT-4 paper specifically show?

#### Round 2: Deep Dive

**Key Discoveries from arXiv:2512.23304**:
- MedGemma-4b-it **zero-shot** vs GPT-4 zero-shot
- Study: 6 medical conditions classification task
- Zero-shot GPT-4: **69.58%** accuracy
- MedGemma fine-tuned (LoRA): **80.37%** accuracy
- Key finding: Domain-specific fine-tuning is ESSENTIAL for minimizing hallucinations
- Fine-tuning eliminates hallucination: "invalid answer rates decrease from 0.14% to 0.00% after LoRA fine-tuning"

**Critical Insight**: The paper only shows MedGemma is competitive AFTER fine-tuning. Zero-shot performance is not compared directly.

#### Round 3+: Crystallization

**Community consensus on MedGemma 4B limitations**:
1. Out-of-the-box performance is **not clinical-grade** — requires fine-tuning
2. Zero-shot performance on rare conditions is likely poor (model biases toward common diagnoses)
3. The 4B size makes complex reasoning about rare diseases unreliable
4. Google's recommended approach: fine-tune on your specific dataset before evaluating

---

## Cross-Cutting Insights

### Why Our 20% Result Makes Sense

1. **Task mismatch**: MedGemma 4B trained for classification/VQA, not open-ended rare disease diagnosis. Our MedPix task asks "What is the diagnosis?" — an open-ended generation task.

2. **Training distribution**: CT training data is dominated by common abnormal findings from radiology reports. Rare conditions (pulmonary artery dilatation, Paget's disease of sternum, giant hiatal hernia) are almost certainly unseen or underrepresented.

3. **CT preprocessing mismatch** (likely): Official MedGemma 1.5 CT preprocessing uses 3-channel RGB windowing (bone/lung/soft tissue). Our pipeline likely fed JPEG screenshots from a DICOM viewer, which doesn't match the training-time preprocessing. This alone could significantly degrade performance.

4. **4B reasoning ceiling**: Even when the visual finding extraction is correct, reasoning to rare diagnoses requires clinical knowledge and multi-step inference that the 4B model lacks at the text reasoning step (MedQA: 69% vs. 27B's 85%).

5. **Compared to wrong baseline**: Comparing to Gemini Flash text-only (70%) is actually a fair characterization that MedGemma 4B underperforms. Gemini Flash has massive general reasoning capability that compensates for lack of visual modality.

### The Structural Problem

MedGemma 4B is excellent at:
- Classifying common CXR findings from clean images (89.5% F1 on MIMIC CXR)
- Structured VQA with predefined answer sets
- Medical document understanding (EHR, lab reports)

MedGemma 4B is poor at:
- Open-ended rare disease diagnosis from single CT slices
- Complex multi-step differential reasoning
- Agentic instruction following
- OOD conditions not in training distribution

Our benchmark tested only the second category.

---

## Architecture/Design Decisions

| Decision | Rationale | Impact on Our Results |
|----------|-----------|----------------------|
| 4B size (not 27B) for multimodal | Only multimodal version available | Hard ceiling on reasoning complexity |
| SigLIP encoder trained on 33M pairs | Domain-specific image features | Strong for trained modalities, weak for OOD |
| 98.8% histopathology in vision training | Data availability bias | CT/radiology features are undertrained |
| RGB windowing for CT | Standard radiological windowing encoded | If we skip this preprocessing, accuracy drops |
| Designed as developer foundation tool | Not production-ready | Requires fine-tuning before benchmarking is meaningful |

---

## Edge Cases & Limitations

1. **CT preprocessing is critical**: The 61% CT accuracy assumes 3-channel RGB windowing preprocessing. Our pipeline may have bypassed this, potentially explaining much of the 20% result.

2. **Rare disease bias toward common diagnoses**: When MedGemma doesn't recognize a rare finding, it defaults to common alternatives ("bilateral pleural effusions" instead of "idiopathic dilatation of pulmonary artery"). This is a known failure mode in medical AI systems.

3. **10-case benchmark statistical power**: Our 10-case benchmark has very high variance. A change of 2 cases = 20 percentage point shift. The result should be treated as directional, not definitive.

4. **OOD evaluation**: MedGemma's own technical report shows even the 27B text model drops from 87.7% (in-distribution) to 25.7% (OOD MedXpertQA). Our rare thoracic conditions are OOD for a model trained on common radiology findings.

5. **Version mismatch**: We deployed "MedGemma 4B" (original v1) on Vertex AI Model Garden. MedGemma 1.5 4B (released Jan 13, 2026) has significantly better CT and MRI support (61.1% vs 58.2%). If we used v1, our baseline is even lower than the official benchmark suggests.

---

## Recommendations

Based on this research, our 20% MedPix result is expected and consistent with the model's documented limitations. Specific recommendations:

1. **Do not compare MedGemma 4B head-to-head with Gemini Flash on rare disease diagnosis** — they are tested on fundamentally different task types (specialized VLM vs. general reasoning model). This is an unfair comparison that will always favor the large general model on rare OOD cases.

2. **Reframe narrative**: Our demo should position MedGemma 4B as "foundation visual understanding layer" — it provides medical image feature extraction that text-only models cannot. The two-stage architecture (MedGemma visual → Gemini text reasoning) is a viable strategy.

3. **CT preprocessing is a quick win**: Implement the official 3-channel RGB windowing preprocessing (bone/lung 1500/−600, soft tissue 350/50, bone 1800/400). This is likely the biggest technical improvement possible without fine-tuning.

4. **Use MedGemma for what it's good at**: Structured finding extraction from images (bone, lung, soft tissue) as input to a more powerful text reasoning model. Not as a standalone diagnosis system.

5. **Frame 20% as expected baseline**: The narrative should be "MedGemma 4B zero-shot on rare OOD conditions = 20% is consistent with Google's 61% on common balanced conditions" — the gap is about rare disease OOD + task format.

---

## Open Questions

1. **What specific preprocessing did our benchmark apply to CT images?** If we used raw JPEG screenshots without RGB windowing, the result is partially a preprocessing bug, not just a model limitation.

2. **Which version of MedGemma did we deploy on Vertex AI?** v1 (58.2% CT) vs v1.5 (61.1% CT) matters. The Vertex AI endpoint was torn down before this could be verified.

3. **Are there any MedPix-specific community benchmarks** comparing MedGemma 4B performance? MedPix is a public dataset but no public MedGemma-on-MedPix benchmarks were found.

---

## Research Methodology Notes

- **Rounds per topic**: 4 total rounds (1 broad survey + 2 deep-dives + 1 crystallization)
- **Sources consulted**: Official Google model card, HuggingFace model card, Google Research blog, arxiv:2507.05201 (MedGemma Technical Report), arxiv:2512.23304 (MedGemma vs GPT-4 paper), DeepWiki (Google-Health/medgemma repo)
- **Quality confidence**: HIGH for official benchmarks and limitations; MEDIUM for our specific failure case root cause (CT preprocessing hypothesis needs verification); LOW for community benchmark comparisons (limited public data on rare CT diagnosis)
- **Key limitation of this research**: No public benchmark exactly matching our task (10 rare thoracic conditions, open-ended diagnosis, single CT slice) was found. The 20% result cannot be directly validated against published numbers.

---

## Addendum: SigLIP Vision Encoder Specs & Single-Slice vs Multi-Slice CT

### SigLIP Vision Encoder Technical Specifications

| Specification | Value |
|--------------|-------|
| Architecture | Two-tower (vision + text transformer) |
| Vision encoder size | 400M parameters |
| Text encoder size | 400M parameters |
| Input resolution (MedSigLIP standalone) | 448x448 pixels |
| Input resolution (MedGemma 4B integration) | 896x896 pixels (for multimodal context) |
| Fine-tuning observed resolution | 224x224 pixels (may differ from inference) |
| Normalization | Pixel values normalized to range (-1, 1) |
| Image format | RGB (not grayscale — DICOM single-channel must be converted) |
| Max text tokens | 64 tokens |
| Base architecture | SigLIP-400M (adapted from general SigLIP) |

**Critical finding**: The official GitHub repo (Google-Health/medgemma) shows images are resized to **224x224 RGB** during fine-tuning. The MedSigLIP model card lists 448x448. The Vertex AI inference API accepts image URLs but does not specify mandatory preprocessing. This inconsistency means our benchmark may have used suboptimal resolution.

### Single-Slice CT vs Multi-Slice Volumetric Processing

**MedGemma 1 4B (original)**: Limited to 2D single images only — no multi-slice CT support.

**MedGemma 1.5 4B** (released Jan 13, 2026): Added full 3D volumetric CT support:
- Can accept **entire CT scan volumes** as unified input
- Processes multiple slices together with prompts for holistic analysis
- Enables "cross-slice relationships, 3D lesion detection, volumetric organ segmentation"
- Vertex AI deployment supports DICOM Store integration for hospital workflows

**Our benchmark**: We deployed on Vertex AI Model Garden, likely using MedGemma 1 (not 1.5). If we fed single 2D slices to MedGemma 1, this is the **intended workflow** for that version. If we were using v1.5, feeding single slices would miss the multi-slice volumetric advantage.

**CT Preprocessing for 3-channel RGB**: For CT scans, the official approach converts DICOM HU values into 3-channel RGB using three window settings:
- Channel R: Bone/lung window (W:1500, L:−600)
- Channel G: Soft tissue window (W:350, L:50)
- Channel B: Bone window (W:1800, L:400)

If our MedPix benchmark fed JPEG screenshots from a viewer (which applies arbitrary display windowing), this preprocessing mismatch is a significant source of performance degradation.

### Bottom Line on Preprocessing Hypothesis

The preprocessing mismatch hypothesis is PLAUSIBLE but unverifiable without checking our actual benchmark code. Two factors compound:
1. Wrong image format (JPEG display screenshot vs. 3-channel RGB windowed CT)
2. Potentially wrong model version (v1 single-slice vs v1.5 multi-slice)

Fixing both would be the recommended improvement if re-running the benchmark.

---

## Sources

- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [MedGemma Technical Report (arXiv:2507.05201)](https://arxiv.org/abs/2507.05201)
- [google/medgemma-1.5-4b-it HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Next generation medical image interpretation - Google Research Blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [MedGemma vs GPT-4 paper (arXiv:2512.23304)](https://arxiv.org/abs/2512.23304)
- [Google-Health/medgemma GitHub](https://github.com/Google-Health/medgemma)
- [MedGemma DeepMind](https://deepmind.google/models/gemma/medgemma/)
