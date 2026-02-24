# MedGemma 4B Imaging Findings Benchmark — Results & Improvement Plan

**Date**: 2026-02-24
**Run directory**: `runs/medpix-multimodal-findings_only-20260223-222130/`
**Dataset**: MedPix 2.0, 20 cases stratified across 5 body regions

## Executive Summary

MedGemma 4B (v1.5, Vertex AI) achieved **25% "good" findings** (LLM judge) on a 20-case multi-region radiology benchmark, compared to **43.8%** for Gemini Flash baseline. The model can identify large, obvious abnormalities but consistently misses subtle secondary findings and fails on Head imaging. This document captures root causes and actionable improvement paths.

## Benchmark Design

### Dataset
- **Source**: MedPix 2.0 (671 cases, 2,050 images)
- **Sampling**: Stratified — 4 cases per body region (Head, Thorax, Abdomen, Spine/MSK, Urinary)
- **Builder script**: `scripts/build_medpix_benchmark_20.py`
- **Benchmark file**: `data/benchmark/medpix_multiregion_20.json`

### Evaluation Metrics
1. **LLM Judge** (Gemini 3 Pro): Rates findings as good/partial/poor based on primary abnormality identification
2. **Key Finding Rate**: Whether primary abnormality was correctly identified
3. **ROUGE-L Recall**: Lexical overlap with gold standard findings
4. **ROUGE-L Precision/F1**: Additional lexical metrics

### Infrastructure
- **MedGemma 4B**: Vertex AI Model Garden, `google/medgemma-1.5-4b-it`, 1x L4 GPU, bf16
- **Gemini Flash**: Google AI Studio, `gemini-3-flash-preview`
- **LLM Judge**: Gemini 3 Pro via `scripts/rescore_simple.py`

## Results

### Overall

| Metric | MedGemma 4B | Gemini Flash | Delta |
|--------|-------------|-------------|-------|
| LLM Judge (good) | 25.0% (4/16) | 43.8% (7/16) | -18.8pp |
| LLM Judge (partial) | 0.0% | 6.2% | -6.2pp |
| LLM Judge (poor) | 75.0% (12/16) | 50.0% (8/16) | +25pp |
| Key Finding Rate | 25.0% | 50.0% | -25pp |
| ROUGE-L Recall | 0.219 | 0.310 | -0.091 |
| ROUGE-L Precision | 0.082 | 0.069 | +0.013 |
| Avg Latency (ms) | 24,509 | 10,652 | +130% |
| Total Cost | $0.157 | $0.114 | +38% |

**Note**: 4/20 judge calls per model returned "error" (Gemini API timeout), so judge metrics are based on n=16.

### By Body Region

| Region | MG good/judged | GM good/judged | MG ROUGE-R | GM ROUGE-R |
|--------|---------------|---------------|------------|------------|
| Head | **0/2** | 2/3 | 0.239 | 0.373 |
| Thorax | 1/3 | 1/3 | 0.128 | 0.241 |
| Abdomen | 1/4 | 1/3 | 0.187 | 0.290 |
| Spine/MSK | 1/3 | 2/3 | 0.253 | 0.278 |
| Urinary | 1/4 | 1/4 | 0.287 | 0.369 |

**Weakest region**: Head (0/2 good) and Thorax (lowest ROUGE-R 0.128).
**Closest to Gemini**: Spine/MSK (ROUGE-R gap only 0.025).

### MedGemma "Good" Cases (4/16)

| UID | Region | Key Finding |
|-----|--------|-------------|
| MPX2015 | Thorax | Large opacity + pleural effusion (Hiatal Hernia) |
| MPX1289 | Abdomen | Large fluid collection (Ceco-Abdominal Fistula) |
| MPX1227 | Spine/MSK | Dense mass in distal femur (Parosteal Osteosarcoma) |
| MPX1012 | Urinary | Large heterogeneous adnexal mass (Ovarian Torsion) |

**Pattern**: MedGemma succeeds on **large, single-abnormality cases** with high-contrast lesions.

### MedGemma Failure Patterns

1. **Missed secondary findings**: Even when primary abnormality is detected, associated findings (effusions, lymphadenopathy, displacement) are consistently missed
2. **Head imaging weakness**: 0/2 good — model struggles with intracranial pathology on CT/MRI
3. **Low ROUGE recall everywhere**: Best region (Urinary, 0.287) still well below Gemini Flash (0.369)
4. **Verbose but imprecise**: ROUGE precision (0.082) suggests model generates text that doesn't match gold standard terminology

## Root Cause Analysis

### 1. Image-Text Ordering Bug (Fixed)
- **Bug**: `vertex_medgemma.py` sent `[image, text]` but MedGemma expects `[text, image]`
- **Fix**: Swapped to `[text, image]` order (commit `bdfea1c`)
- **Impact**: This fix was already applied before the 20-case benchmark

### 2. Model Architecture Limitations (4B)
- MedGemma 4B is a relatively small multimodal model
- Limited capacity for complex multi-finding radiology reports
- Works best as a "screening" tool (is there a mass? yes/no) rather than comprehensive reporting

### 3. Prompt Design
- Current prompt asks for detailed systematic findings
- 4B model may perform better with focused, targeted questions
- No system message optimization for radiology-specific behavior

### 4. Single Image Input
- Many MedPix cases have multiple images (different views/sequences)
- Benchmark only sends first image — model misses findings visible on other views
- Gemini Flash likely benefits from stronger single-image reasoning

## Improvement Recommendations

### High Priority (Expected Impact: +10-15pp)

#### A. Prompt Engineering
- **Focused prompting**: Instead of "describe ALL findings", ask targeted questions per region:
  - "Is there a mass? Describe location and size"
  - "Are there any effusions?"
  - "Describe bone density abnormalities"
- **Few-shot examples**: Include 1-2 example findings in the prompt
- **Region-specific prompts**: Different templates for chest X-ray vs brain MRI vs abdominal CT

#### B. Multi-Image Support
- Send all available images for each case (up to 4 per `--limit-mm-per-prompt=image=4`)
- Many MedPix cases have 2-3 images that show different aspects of pathology
- Expected improvement on cases where key finding is only visible on non-primary image

#### C. Two-Stage Findings Pipeline
- **Stage 1 (MedGemma 4B)**: Identify primary abnormality and location
- **Stage 2 (Gemini Flash)**: Elaborate on secondary findings using MedGemma's initial read
- Mirrors the successful two-stage architecture from criterion-level evaluation (87.5% accuracy)

### Medium Priority (Expected Impact: +5-10pp)

#### D. Model Version / Size
- Monitor MedGemma releases — v2.0 may have improved multimodal capabilities
- Consider MedGemma 27B for text-only findings refinement (if image features extracted separately)
- Evaluate Google's upcoming medical-specific vision models

#### E. Temperature and Sampling
- Current: `temperature=0.0` (greedy decoding)
- Experiment with `temperature=0.3` for more diverse findings generation
- Self-consistency: Generate N findings and take consensus

#### F. Post-Processing
- Extract structured findings from free-text output
- Map to standard radiology vocabulary (RadLex)
- Score against structured gold standard rather than ROUGE

### Low Priority / Research

#### G. Fine-tuning (if MedGemma supports LoRA)
- Fine-tune on MedPix gold findings (671 cases available)
- Would require MedGemma LoRA adapter support on Vertex AI

#### H. Retrieval-Augmented Generation
- Retrieve similar cases from MedPix during inference
- Use retrieved findings as few-shot context

## Benchmark Infrastructure Notes

### What Worked
- Stratified multi-region sampling gives balanced evaluation
- LLM-as-judge (Gemini Pro) provides clinically meaningful quality assessment beyond ROUGE
- Offline rescoring (`rescore_simple.py`) recovers from API failures
- Run artifact persistence (`raw_responses.json`) enables re-evaluation without re-running models

### What Needs Improvement
- **Judge reliability**: 20% error rate on judge calls — need retry logic or fallback judge
- **Sample size**: n=20 (16 judged) is too small for statistical significance per region
- **Scoring latency**: Judge scoring took multiple attempts due to Gemini API instability
- **Missing per-image analysis**: No breakdown of which image modality (CT/MRI/XR) performs best

### Recommended Next Benchmark
- **Scale to 50 cases** (10 per region) for better statistical power
- **Add multi-image cases** to test image fusion capability
- **Add modality stratification** (CT vs MRI vs X-ray) as independent variable
- **Implement judge retry with exponential backoff** (max 3 retries per case)
- **A/B test prompt variants** (focused vs comprehensive) on same cases

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/build_medpix_benchmark_20.py` | Stratified 20-case benchmark builder |
| `scripts/run_medpix_benchmark.py` | Main benchmark runner with LLM judge |
| `scripts/rescore_simple.py` | Offline rescoring from raw responses |
| `scripts/analyze_findings_results.py` | Per-case/per-region breakdown analysis |
| `configs/medpix_bench_multiregion_20.yaml` | 20-case benchmark config |
| `data/benchmark/medpix_multiregion_20.json` | 20-case benchmark dataset |
| `runs/medpix-multimodal-findings_only-20260223-222130/` | Full run artifacts |
