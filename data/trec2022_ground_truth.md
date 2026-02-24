Directory structure:
└── trec2022_ground_truth/
    ├── README.md
    ├── expected_outcomes.json
    ├── patient.jsonl
    └── qrels.tsv

================================================
FILE: trial_matching_agent/data/trec2022_ground_truth/README.md
================================================
# TREC 2022 Ground Truth — Patient trec-20226 (Malignant Pleural Mesothelioma)

Official ground-truth relevance judgments from the **TREC 2022 Clinical Trials Track**
for patient `trec-20226`, a 61-year-old male with malignant pleural mesothelioma.

## Source

- **TREC 2022 Clinical Trials Track**: https://www.trec-cds.org/2022.html
- **TrialGPT dataset**: https://github.com/ncbi-nlp/TrialGPT/tree/main/dataset/trec_2022
- **Trial criteria corpus**: `trial_info.json` (448,631 trials from ClinicalTrials.gov)
- **Qrels**: `trec_2022/qrels/test.tsv` — official NIST relevance judgments

## Dataset Files

| File | Format | Size | Description |
|------|--------|------|-------------|
| `patient.jsonl` | JSONL (1 row) | 2.7 KB | Patient EHR text, numbered sentences, structured profile |
| `corpus.jsonl` | JSONL (647 rows) | 1.7 MB | All judged clinical trials with full inclusion/exclusion criteria |
| `qrels.tsv` | TSV | 16 KB | Official TREC relevance judgments: `query-id → corpus-id → score` |
| `expected_outcomes.json` | JSON | 50 KB | Grouped trial lists by label, patient summary, scoring expectations |

## Connection Logic

```
patient.jsonl                    qrels.tsv                      corpus.jsonl
┌──────────────────┐          ┌──────────────────────┐       ┌──────────────────────┐
│_id: "trec-20226" │◄────────►│query-id: trec-20226  │       │_id: "NCT00002622"    │
│text: "A 61-year… │          │corpus-id: NCT00002622│◄─────►│brief_title: "Talc…"  │
│structured_profile│          │score: 2              │       │inclusion_criteria: … │
│  age: 61         │          │                      │       │exclusion_criteria: … │
│  diagnosis: meso…│          │corpus-id: NCT00079235│       │brief_summary: …      │
└──────────────────┘          │score: 1              │       └──────────────────────┘
                              │                      │        (647 trial records)
                              │corpus-id: NCT00000644│
                              │score: 0              │
                              │ …(647 total lines)   │
                              └──────────────────────┘

  expected_outcomes.json
  ┌─────────────────────────────────────────────┐
  │ References patient._id and corpus._id       │
  │ Groups trials into:                         │
  │   eligible_trials[]    → label 2 (118 NCTs) │
  │   partial_trials[]     → label 1  (20 NCTs) │
  │   excluded_trials[]    → label 0 (509 NCTs) │
  └─────────────────────────────────────────────┘
```

### Join Keys

- **Patient → Qrels**: `patient._id == qrels.query-id`
- **Qrels → Corpus**: `qrels.corpus-id == corpus._id`
- **Expected Outcomes → Corpus**: `expected_outcomes.*.nct_id == corpus._id`

## Label Distribution

| Score | Label | Count | Meaning |
|-------|-------|-------|---------|
| **2** | Eligible | 118 | Patient meets trial criteria; disease/condition match confirmed |
| **1** | Partially relevant | 20 | Related disease/procedure but some criteria unmet or unclear |
| **0** | Not relevant / Excluded | 509 | Patient does not match the trial's target population |
| | **Total** | **647** | |

## Patient Summary

| Field | Value |
|-------|-------|
| **ID** | trec-20226 |
| **Age/Sex** | 61-year-old male |
| **Primary Diagnosis** | Malignant pleural mesothelioma (epithelioid type) |
| **Key Biopsy Finding** | Epithelioid-type cells with very long microvilli |
| **Imaging** | CT: left-sided pleural effusion, nodular pleural thickening |
| **Thoracentesis** | Bloody pleural fluid |
| **Smoking** | 2 packs/day × 30 years (60 pack-years, current smoker) |
| **Comorbidities** | Hypertension, hypercholesterolemia, peptic ulcer disease |
| **Vitals** | Normal |

## Trial Categories in Ground Truth

### Label 2 — Eligible (118 trials)

Dominant categories:
- **Mesothelioma treatment trials** — chemotherapy (pemetrexed + cisplatin), immunotherapy, gene therapy
- **Pleural effusion management** — pleurodesis, indwelling pleural catheters, drainage studies
- **Pleural disease diagnostics** — thoracoscopy, biopsy methods, biomarker studies
- **Thoracic oncology** — lung cancer screening, breath tests applicable to this patient

Example eligible trials:
| NCT ID | Trial Name |
|--------|-----------|
| NCT00005636 | Cisplatin ± Pemetrexed in Malignant Mesothelioma |
| NCT01766739 | Intra-pleural GL-ONC1 (Vaccinia Virus) in Lung Cancer |
| NCT03507452 | Thorium-227 in Advanced Malignant Pleural Mesothelioma |
| NCT04322136 | IPC Plus Talc vs VATS for Malignant Pleural Effusion |
| NCT00896285 | First Therapeutic Intervention in Malignant Pleural Effusion |

### Label 1 — Partially Relevant (20 trials)

Trials that address related conditions but have criteria mismatches:
| NCT ID | Trial Name | Why Partial |
|--------|-----------|-------------|
| NCT00079235 | CCI-779 in Stage IIIB/IV NSCLC | Requires confirmed NSCLC (patient has mesothelioma) |
| NCT03550027 | IPC for Trapped Lung | Pleural disease overlap but different mechanism |
| NCT04300244 | Nivolumab + Ipilimumab ± UV1 in Mesothelioma | Second-line (treatment history unclear) |
| NCT03583931 | Fibrinolytic Therapy for Parapneumonic Effusion | Effusion overlap but different etiology |

### Label 0 — Excluded (509 trials)

Trials targeting completely unrelated conditions (HIV, diabetes, pediatric diseases, etc.).

## Usage

```python
import json
from collections import Counter

GT = "trial_matching_agent/data/trec2022_ground_truth"

# Load all files
with open(f"{GT}/patient.jsonl") as f:
    patient = json.loads(f.readline())

trials = {}
with open(f"{GT}/corpus.jsonl") as f:
    for line in f:
        t = json.loads(line)
        trials[t["_id"]] = t

qrels = {}
with open(f"{GT}/qrels.tsv") as f:
    next(f)  # skip header
    for line in f:
        qid, cid, score = line.strip().split("\t")
        qrels.setdefault(qid, {})[cid] = int(score)

with open(f"{GT}/expected_outcomes.json") as f:
    expected = json.load(f)

# Iterate over patient-trial pairs with labels
patient_qrels = qrels[patient["_id"]]
labels = Counter(patient_qrels.values())
print(f"Patient {patient['_id']}: {len(patient_qrels)} trials")
print(f"  Eligible: {labels[2]}, Partial: {labels[1]}, Excluded: {labels[0]}")

for nct_id, score in list(patient_qrels.items())[:5]:
    trial = trials[nct_id]
    label_text = {0: "excluded", 1: "partial", 2: "eligible"}[score]
    print(f"  {nct_id} ({label_text}): {trial['brief_title'][:60]}")
```

## Relationship to Demo Dataset

The files in the **parent directory** (`trec2022_nsclc_*.{jsonl,tsv,json}`) contain a
curated 6-trial subset designed for pipeline demonstration with balanced labels (2/2/2).
This `trec2022_ground_truth/` folder contains the **complete official TREC judgments**
(647 trials) for the same patient.

| Aspect | Demo Dataset (parent dir) | Ground Truth (this dir) |
|--------|--------------------------|------------------------|
| Location | `data/trec2022_nsclc_*` | `data/trec2022_ground_truth/` |
| Patient | trec2022-topic1 (synthetic) | trec-20226 (TREC official) |
| Trials | 6 (curated) | 647 (all judged) |
| Labels | 2/2/2 balanced | 118/20/509 natural |
| Purpose | Pipeline demo & walkthrough | Evaluation & benchmarking |
| Criteria | Hand-written expected matching | Official NIST qrels |



================================================
FILE: trial_matching_agent/data/trec2022_ground_truth/expected_outcomes.json
================================================
{
  "_metadata": {
    "description": "TREC 2022 Clinical Trials Track ground truth for patient trec-20226 (malignant pleural mesothelioma)",
    "patient_file": "patient.jsonl",
    "corpus_file": "corpus.jsonl",
    "qrels_file": "qrels.tsv",
    "join_key": "trec-20226 (query-id) \u2192 NCT* (corpus-id)",
    "label_schema": {
      "2": "eligible \u2014 patient meets key inclusion criteria and no exclusion criteria are triggered",
      "1": "partially relevant \u2014 trial addresses related disease/procedure but some criteria unmet or unclear",
      "0": "not relevant / excluded \u2014 patient does not match the trial's target population"
    },
    "source": "TREC 2022 Clinical Trials Track (https://www.trec-cds.org/2022.html)",
    "trialgpt_repo": "https://github.com/ncbi-nlp/TrialGPT/tree/main/dataset/trec_2022",
    "label_distribution": {
      "eligible (2)": 118,
      "partial (1)": 20,
      "excluded (0)": 509,
      "total": 647
    }
  },
  "patient_summary": {
    "patient_id": "trec-20226",
    "age": 61,
    "sex": "male",
    "primary_diagnosis": "Malignant pleural mesothelioma (epithelioid type)",
    "key_findings": [
      "61-year-old male, heavy smoker (60 pack-years)",
      "Nonproductive cough and progressive dyspnea",
      "Left-sided pleural effusion with nodular pleural thickening on CT",
      "Bloody pleural fluid on thoracentesis",
      "Biopsy: epithelioid-type cells with very long microvilli (diagnostic of mesothelioma)",
      "Comorbidities: hypertension, hypercholesterolemia, peptic ulcer disease"
    ],
    "clinical_reasoning": "The biopsy finding of 'epithelioid-type cells with very long microvilli' is the hallmark histological feature of MALIGNANT MESOTHELIOMA (epithelioid subtype). The nodular pleural thickening and bloody effusion are classic imaging/clinical findings. This patient is eligible for: (1) mesothelioma treatment trials, (2) pleural effusion management trials, (3) pleural disease diagnostic studies, and (4) general thoracic oncology trials. Trials requiring confirmed NSCLC, specific mutations, or unrelated diseases should be labeled 0."
  },
  "eligible_trials": [
    {
      "nct_id": "NCT00002622",
      "trial_name": "Talc in Treating Patients With Malignant Pleural Effusion",
      "diseases": [
        "Metastatic Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00002872",
      "trial_name": "Bleomycin, Doxycycline, or Talc in Treating Patients With Malignant Pleural Effusions",
      "diseases": [
        "Metastatic Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00005636",
      "trial_name": "Cisplatin With or Without Pemetrexed Disodium in Treating Patients With Malignant Mesothelioma of the Pleura That Cannot be Removed by Surgery",
      "diseases": [
        "Malignant Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00019825",
      "trial_name": "Decitabine in Treating Patients With Unresectable Lung or Esophageal Cancer or Malignant Mesothelioma of the Pleura",
      "diseases": [
        "Esophageal Cancer",
        "Lung Cancer",
        "Malignant Mesothelioma",
        "Metastatic Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00037817",
      "trial_name": "Phase I Study of Gene Induction Mediated by Sequential Decitabine/Depsipeptide Infusion With or Without Concurrent Celecoxib in Subjects With Pulmonary and Pleural Malignancies",
      "diseases": [
        "Advanced Esophageal Cancers",
        "Primary Small Cell Lung Cancers",
        "Non-Small-Cell Lung Carcinoma",
        "Pleural Mesotheliomas",
        "Cancers of Non-thoracic Origin With Metastases to the Lungs or Pleura"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00042770",
      "trial_name": "Standard Chest Tube Compared With a Small Catheter in Treating Malignant Pleural Effusion in Patients With Cancer",
      "diseases": [
        "Metastatic Cancer",
        "Pulmonary Complications"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00094978",
      "trial_name": "Depsipeptide/Flavopiridol Infusion for Cancers of the Lungs, Esophagus, Pleura, Thymus or Mediastinum",
      "diseases": [
        "Carcinoma",
        "Small Cell",
        "Carcinoma",
        "Non-Small-Cell Lung",
        "Esophageal Neoplasms",
        "Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00114205",
      "trial_name": "Surgery and Intrapleural Docetaxel in Treating Patients With Malignant Pleural Effusion",
      "diseases": [
        "Metastatic Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00165555",
      "trial_name": "Pleurectomy/Decortication Followed by Intrathoracic/Intraperitoneal Heated Cisplatin for Malignant Pleural Mesothelioma",
      "diseases": [
        "Pleural Mesothelioma",
        "Malignant Pleural Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00166894",
      "trial_name": "Use of the Triggering Receptor Expressed on Myeloid Cells-1 (TREM-1) in the Diagnosis of Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00172835",
      "trial_name": "Use of Procalcitonin in the Diagnosis of Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00227630",
      "trial_name": "Pemetrexed Disodium and Cisplatin Followed By Surgery and Radiation Therapy in Treating Patients With Malignant Pleural Mesothelioma",
      "diseases": [
        "Malignant Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00242723",
      "trial_name": "Evaluation of Cell Changes in Blood and Tissue in Cancers of the Lung, Esophagus and Lung Lining",
      "diseases": [
        "Malignant Pleural Mesotheliomas NOS",
        "Esophageal Cancers NOS",
        "Lung Cancer NOS",
        "Thoracic Cancers",
        "Cancers of Non Thoracic Origin With Metastases to the Lungs or Pleura"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00313066",
      "trial_name": "Comparison the Level of CTGF Protein and Related Cytokine in Pleural Effusion",
      "diseases": [
        "Tuberculosis",
        "Tuberculous Pleurisy",
        "Malignant Pleural Effusion",
        "Empyema"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00316134",
      "trial_name": "Multiple Biomarkers in Undiagnosed Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00430664",
      "trial_name": "A Comparative Study of the Safety and Efficacy of Face Talc Slurry and Iodopovidone for Pleurodesis",
      "diseases": [
        "Malignant Pleural Effusions",
        "Recurrent Pleural Effusions",
        "Primary Spontaneous Pneumothorax",
        "Secondary Spontaneous Pneumothorax"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00469196",
      "trial_name": "Tomotherapy Treatment for Mesothelioma",
      "diseases": [
        "Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00473291",
      "trial_name": "Vibration Response Imaging (VRI) in Management and Evaluation in Patients With Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00504738",
      "trial_name": "Lung Disease Collection (Qatar): Evaluation of the Lungs of Individuals With Lung Disease",
      "diseases": [
        "Lung Disease",
        "Chronic Obstructive Pulmonary Disease (COPD)",
        "Asthma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00509041",
      "trial_name": "Dasatinib in Treating Patients With Previously Treated Malignant Mesothelioma",
      "diseases": [
        "Malignant Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00637078",
      "trial_name": "STITCH2 (Simplified Therapeutic Intervention to Control Hypertension and Hypercholesterolemia)",
      "diseases": [
        "Hypertension",
        "Hypercholesterolemia"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00637676",
      "trial_name": "Tunneled Pleural Catheter in Partially Entrapped Lung",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00639067",
      "trial_name": "Breath Test for Early Detection of Lung Cancer",
      "diseases": [
        "Lung Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00644319",
      "trial_name": "Ibuprofen or Morphine in Treating Pain in Patients Undergoing Pleurodesis for Malignant Pleural Effusion",
      "diseases": [
        "Metastatic Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00690261",
      "trial_name": "The Impact of M1/M2 Tumor Associated Macrophage (TAM) Polarization on Cancer Progression and Prognosis Prediction",
      "diseases": [
        "Tumor",
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00715611",
      "trial_name": "Pleurectomy/Decortication (Neo) Adjuvant Chemotherapy and Intensity Modulated Radiation Therapy to the Pleura in Patients With Locally Advanced Malignant Pleural Mesothelioma",
      "diseases": [
        "Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00758316",
      "trial_name": "A Prospective, Randomized Controlled Trial for a Rapid Pleurodesis Protocol for the Management of Pleural Effusions",
      "diseases": [
        "Malignant Pleural Effusions"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00789087",
      "trial_name": "Talc Pleurodesis in Patients With Recurrent Malignant Pleural Effusion",
      "diseases": [
        "Recurrent Malignant Pleural Effusion."
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00896285",
      "trial_name": "The First Therapeutic Intervention in Malignant Pleural Effusion Trial",
      "diseases": [
        "Malignant Pleural Effusion",
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT00978939",
      "trial_name": "Impact of Aggressive Versus Standard Drainage Regimen Using a Long Term Indwelling Pleural Catheter",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01105390",
      "trial_name": "AMG 102, Pemetrexed Disodium, and Cisplatin in Treating Patients With Malignant Pleural Mesothelioma",
      "diseases": [
        "Advanced Malignant Mesothelioma",
        "Epithelial Mesothelioma",
        "Recurrent Malignant Mesothelioma",
        "Sarcomatous Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01125124",
      "trial_name": "Safety and Effectiveness Study for Pleurodesis With Silver Nitrate in Malignant Pleural Effusion",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01141946",
      "trial_name": "Pleural Ultrasonography in Lung Cancer",
      "diseases": [
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01193439",
      "trial_name": "Safety of Thoracoscopy in Patients With High Risk",
      "diseases": [
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01196585",
      "trial_name": "Ultrasonography Guided Pleural Biopsy Versus Computed Tomography Guided Pleural Biopsy",
      "diseases": [
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01262612",
      "trial_name": "Cediranib as Palliative Treatment in Patients With Symptomatic Malignant Ascites or Pleural Effusion",
      "diseases": [
        "Malignant Ascites",
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01409551",
      "trial_name": "Video-assisted Hyperthermic Pleural Chemoperfusion vs Talc Pleurodesis for Refractory Malignant Pleural Effusions.",
      "diseases": [
        "Safety of Intervention",
        "Efficacy of Intervention",
        "Cost Effectiveness"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01411202",
      "trial_name": "Effectiveness of Doxycycline for Treating Pleural Effusions Related to Cancer in an Outpatient Population",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01580982",
      "trial_name": "Molecular Analysis of Oncogenes and Resistance Mechanisms in Lung Cancer",
      "diseases": [
        "Lung Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01601444",
      "trial_name": "Detection of Pleural Effusion by Internal Thoracic Impedance Method",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01624090",
      "trial_name": "Mithramycin for Lung, Esophagus, and Other Chest Cancers",
      "diseases": [
        "Lung Cancer",
        "Esophageal Cancer",
        "Mesothelioma",
        "Gastrointestinal Neoplasms",
        "Breast Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01756742",
      "trial_name": "Effects of Respiratory Physiotherapy on Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01766739",
      "trial_name": "Intra-pleural Administration of GL-ONC1, a Genetically Modified Vaccinia Virus, in Patients With Malignant Pleural Effusion: Primary, Metastases and Mesothelioma",
      "diseases": [
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01778270",
      "trial_name": "Not Invasive Monitoring of Pleural Drainage",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01819363",
      "trial_name": "Relationship Between Pleural Elastance and Effectiveness of Pleurodesis on Recurrent Malignant Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01952327",
      "trial_name": "Investigation Into the Automated Drainage of Recurrent Effusions From the Pleural Space in Thoracic Malignancy.",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01952431",
      "trial_name": "Measurement and Evaluation of Total Lung Capacity (TLC) in the Field of Pulmonary Functional Testing (PFT)",
      "diseases": [
        "TLC in Patients With and Without Respiratory System Disease"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT01973985",
      "trial_name": "Using Ultrasound to Predict the Results of Draining Pleural Effusions",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02024113",
      "trial_name": "LC-NMR Study Biomarkers to Detect Lung Cancer",
      "diseases": [
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02045121",
      "trial_name": "Multicentre Study Comparing Indwelling Pleural Catheter With Talc Pleurodesis for Malignant Pleural Effusion Management",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02062632",
      "trial_name": "Doxepin Hydrochloride in Treating Esophageal Pain in Patients With Thoracic Cancer Receiving Radiation Therapy to the Thorax With or Without Chemotherapy",
      "diseases": [
        "Esophageal Carcinoma",
        "Hypopharyngeal Carcinoma",
        "Laryngeal Carcinoma",
        "Lymphoma",
        "Mesothelioma",
        "Metastatic Malignant Neoplasm in the Lung",
        "Metastatic Malignant Neoplasm in the Pleura",
        "Metastatic Malignant Neoplasm in the Spinal Cord",
        "Non-Small Cell Lung Carcinoma",
        "Sarcoma",
        "Small Cell Lung Carcinoma",
        "Thymic Carcinoma",
        "Thymoma",
        "Thyroid Gland Carcinoma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02092155",
      "trial_name": "Biomarker Levels During Indwelling Pleural cAtheter Sample Testing",
      "diseases": [
        "Malignant Pleural Effusions"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02172027",
      "trial_name": "Immunomagnetic Detection of Cancer Cells in Pleural Effusion in Lung Cancer Patients as Additional Staging and Prognostic Tool",
      "diseases": [
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02192138",
      "trial_name": "Pathophysiological Effects of Intrapleural Pressure Changes During Therapeutic Thoracentesis",
      "diseases": [
        "Pleural Effusion",
        "Exudative Pleuritis"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02208895",
      "trial_name": "iSTAT Comparison Study, IRB3785",
      "diseases": [
        "Pleural Effusions"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02227732",
      "trial_name": "A Pilot Study Evaluating the Safety and Effectiveness of a New Pleural Catheter for the Medical Management of Symptomatic, Recurrent, Malignant Pleural Effusions",
      "diseases": [
        "Malignant Pleural Effusions"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02246946",
      "trial_name": "Positive Airway Pressure on Pleural Effusion After Drainage",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02269761",
      "trial_name": "Chest Ultrasound of ER Patients With Cough or SOB",
      "diseases": [
        "Cough",
        "Dyspnea",
        "Wheezing"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02300883",
      "trial_name": "Observational Prospective Analysis of Biological Characteristics of Mesothelioma Patients",
      "diseases": [
        "Malignant Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02320617",
      "trial_name": "Application of Diffusion Weighted MRI Versus CT in Evaluation of the Effect of Treating Lung Cancer",
      "diseases": [
        "Lung Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02414269",
      "trial_name": "Malignant Pleural Disease Treated With Autologous T Cells Genetically Engineered to Target the Cancer-Cell Surface Antigen Mesothelin",
      "diseases": [
        "Malignant Pleural Disease",
        "Mesothelioma",
        "Metastases",
        "Lung Cancer",
        "Breast Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02436850",
      "trial_name": "Bed Side Thoracentesis Among Non-Ventilated Patients With Respiratory Instability",
      "diseases": [
        "Respiratory Instability"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02517749",
      "trial_name": "Out Patient Talc Slurry Via Indwelling Pleural Catheter for Malignant Pleural Effusion Vs Usual Inpatient Management",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02518100",
      "trial_name": "Vital Signs Patch: Automated Monitoring of Vital Signs Measurements in the In-Patient Hospital Setting",
      "diseases": [
        "Automated Measurement of Vital Signs"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02521922",
      "trial_name": "Vital Signs Patch Early Feasibility and Usability Study",
      "diseases": [
        "Vital Signs"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02524470",
      "trial_name": "Vital Signs Patch Early Feasibility and Usability Study v1.0",
      "diseases": [
        "Vital Signs"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02548221",
      "trial_name": "Impact of Large-Volume Pleural Effusions on Heart Function",
      "diseases": [
        "Bilateral Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02583282",
      "trial_name": "A Study to Compare the Efficacy and Safety of Intrapleural Doxycycline Versus Iodopovidone for Performing Pleurodesis in Malignant Pleural Effusion",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02602119",
      "trial_name": "Intraoperative Imaging of Pulmonary Nodules by OTL38",
      "diseases": [
        "Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02613299",
      "trial_name": "Surgery for Mesothelioma After Radiation Therapy SMART for Resectable Malignant Pleural Mesothelioma",
      "diseases": [
        "Mesothelioma",
        "Solitary Fibrous Tumor of the Pleura"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02613312",
      "trial_name": "Chemotherapy Followed by Surgery and Neoadjuvant Hemothoracic Intensity Modified Radiation Therapy (IMRT) for Patients With Malignant Pleural Mesothelioma",
      "diseases": [
        "Mesothelioma",
        "Solitary Fibrous Tumor of the Pleura"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02627274",
      "trial_name": "A Study Evaluating Safety, Pharmacokinetics, and Therapeutic Activity of RO6874281 as a Single Agent (Part A) or in Combination With Trastuzumab or Cetuximab (Part B or C)",
      "diseases": [
        "Solid Tumor",
        "Breast Cancer",
        "Cancer of Head and Neck"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02648763",
      "trial_name": "Staging Procedures to Diagnose Malignant Pleural Mesothelioma",
      "diseases": [
        "Mesothelioma",
        "Solitary Fibrous Tumor of the Pleura"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02649894",
      "trial_name": "Safety and Effectiveness of a New Pleural Catheter for Symptomatic, Recurrent, MPEs Versus Approved Pleural Catheter",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02677883",
      "trial_name": "Impact of Pleural Manometry on Chest Discomfort After Therapeutic Thoracentesis",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02724462",
      "trial_name": "Trial of An Innovative Smartphone Intervention for Smoking Cessation",
      "diseases": [
        "Smoking"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02809872",
      "trial_name": "Ultrasound Estimation of Pleural Effusion in the Sitting Patients",
      "diseases": [
        "Pleural Effusion Disorder",
        "Thoracic Ultrasound"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02825095",
      "trial_name": "Management of Malignant Pleural Effusion - Indwelling Pleural Catheter or Talc Pleurodesis",
      "diseases": [
        "Pleural Effusion",
        "Malignant",
        "Lung Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02851927",
      "trial_name": "Mini Thoracoscopy vs Semirigid Thoracoscopy in Exudative Pleural Effusions",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT02975921",
      "trial_name": "Betadine Pleurodesis Via Tunneled Pleural Catheters",
      "diseases": [
        "Pleural Effusion",
        "Pleurodesis",
        "Malignant Pleural Effusion",
        "Pleural Effusion Due to Congestive Heart Failure",
        "Pleural Effusion in Conditions Classified Elsewhere",
        "Pleural Effusions",
        "Chronic"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03153501",
      "trial_name": "Efficiency and Safety of Pleural Biopsy Methods in the Diagnosis of Pleural Diseases",
      "diseases": [
        "Diagnosis of Pleural Diseases",
        "Medical Thoracoscopy"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03172052",
      "trial_name": "Evaluating Different Modalities for Pleural Adhesiolysis at Assuit University Hospital",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03189108",
      "trial_name": "Collection of Malignant Ascites, Pleural Fluid, and Blood From People With Solid Tumors",
      "diseases": [
        "Ovarian Cancer",
        "Primary Peritoneal Cancer",
        "Fallopian Tube Cancer",
        "Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03270215",
      "trial_name": "The Added Value of CT Scanning in Patients With an Unilateral Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03276715",
      "trial_name": "Prognostic Factors on Malignant Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Malignant",
        "Lung Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03290183",
      "trial_name": "Confocal Laser Endomicroscopy in Pleural Malignancies",
      "diseases": [
        "Pleural Malignant Mesothelioma",
        "Thymoma",
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03319186",
      "trial_name": "EDIT Management Feasibility Trial",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03319472",
      "trial_name": "Using Pleural Effusions to Diagnose Cancer",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03325192",
      "trial_name": "Rapid Pleurodesis Through an Indwelling Pleural Catheter",
      "diseases": [
        "Pleural Effusion",
        "Malignant",
        "Pleurodesis",
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03403855",
      "trial_name": "Rocket\u00ae Pleural Catheters: QOL, Feasibility and Satisfaction in Recurrent MPE Patients",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03414905",
      "trial_name": "Management of Malignant Pleural Effusions Using an Indwelling Tunneled Pleural Catheter and Non-Vacuum Collection System",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03482570",
      "trial_name": "Activity Behaviours in Patients With Malignant Pleural Effusion",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03507452",
      "trial_name": "First-in-human Study of BAY2287411 Injection, a Thorium-227 Labeled Antibody-chelator Conjugate, in Patients With Tumors Known to Express Mesothelin",
      "diseases": [
        "Advanced Recurrent Malignant Pleural Epithelioid Mesothelioma",
        "Advanced Recurrent Malignant Peritoneal Epithelioid Mesothelioma",
        "Advanced Recurrent Serous Ovarian Cancer",
        "Advanced Pancreatic Ductal Adenocarcinoma (Optional",
        "Dose Expansion",
        "Not Initiated)"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03591952",
      "trial_name": "Gravity- Versus Suction-driven Large Volume Thoracentesis",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03678090",
      "trial_name": "The Safety and Efficacy of Fibrinolysis in Patients With an Indwelling Pleural Catheter for Multi-loculated Malignant Pleural Effusion.",
      "diseases": [
        "Pleural Effusion",
        "Cancer",
        "Lung"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03736122",
      "trial_name": "A Study of Syngenon (BSG-001) for Inhalation in Subjects With Malignant Pleural Effusion and/or Malignant Ascites",
      "diseases": [
        "Malignant Pleural Effusion",
        "Malignant Ascites"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03781908",
      "trial_name": "Management of Malignant Pleural Effusion With Indwelling Pleural Catheter Versus Silver Nitrate Pleurodesis",
      "diseases": [
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03831386",
      "trial_name": "Gravity Versus Vacuum Based Indwelling Tunneled Pleural Drainage System",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03907852",
      "trial_name": "Phase 1/2 Trial of Gavo-cel (TC-210) in Patients With Advanced Mesothelin-Expressing Cancer",
      "diseases": [
        "Mesothelioma",
        "Mesothelioma",
        "Malignant",
        "Mesothelioma; Pleura",
        "Mesotheliomas Pleural",
        "Mesothelioma Peritoneum",
        "Cholangiocarcinoma",
        "Cholangiocarcinoma Recurrent",
        "Ovarian Cancer",
        "Non Small Cell Lung Cancer",
        "Non Small Cell Lung Cancer Metastatic",
        "High Grade Ovarian Serous Adenocarcinoma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03922841",
      "trial_name": "Pleural Disease: Phenotypes, Diagnostic Yield and Outcomes",
      "diseases": [
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03973957",
      "trial_name": "Talc Outpatient Pleurodesis With Indwelling Catheter",
      "diseases": [
        "Pleural Effusion",
        "Pleural Diseases",
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT03997669",
      "trial_name": "The Diagnosis and Mechanism of Pleural Effusion",
      "diseases": [
        "Pleural Diseases"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04039126",
      "trial_name": "Comparison of the Effectiveness of Povidone-Iodine Alone to Povidone-Iodine-Tetracycline Combination",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04074902",
      "trial_name": "Role of Chest Sonography in Evaluation of Pleurodesis in Patients With Malignant Pleural Effusion",
      "diseases": [
        "Pleurodesis"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04115111",
      "trial_name": "Diadem to Investigate the Activity and Safety of Durvalumab",
      "diseases": [
        "Pleura Mesothelioma"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04264325",
      "trial_name": "Quantitative Diaphragmatic Ultrasound Evaluation in Pleural Effusions : A Feasability Study",
      "diseases": [
        "Malignant Pleural Effusion",
        "Lung Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04322136",
      "trial_name": "AMPLE-3: IPC Plus Talc vs VATS in Management of Malignant Pleural Effusion",
      "diseases": [
        "Malignant Pleural Effusion",
        "Respiratory Disease",
        "Cancer"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04413292",
      "trial_name": "Survivin and Fibulin-3 in Benign and Malignant Respiratory Diseases",
      "diseases": [
        "Bronchial Neoplasm Benign"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04418804",
      "trial_name": "Diagnosis and Classification of Pleural Diseases Using Ultrasound Channel Data",
      "diseases": [
        "Pneumothorax",
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04420663",
      "trial_name": "Pleural Manometry in Thoracocentesis",
      "diseases": [
        "Thoracocentesis of Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04533854",
      "trial_name": "Investigating Signal Change in Malignant and Non-malignant Pleural Effusions and asCitic Fluid Using fTiR Analysis",
      "diseases": [
        "Pleural Effusion",
        "Ascites"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04670562",
      "trial_name": "Longitudinal Follow up of Patients With Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Malignant",
        "Prognosis"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04683419",
      "trial_name": "Modified Thoracoscopic Pleural Cryobiopsy in Diagnosis of Exudative Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04731129",
      "trial_name": "Mini Invasive Endomicroscopy of the Pleura for Malignancies Diagnosis",
      "diseases": [
        "Pleural Diseases",
        "Pleural Neoplasms"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04761003",
      "trial_name": "Efficacy And Safety Of Thoracoscopic Cryobiopsy In Patients With Undiagnosed Exudative Pleural Effusion",
      "diseases": [
        "Pleura; Exudate"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04781894",
      "trial_name": "Application of Transthoracic Shear-wave Ultrasound Elastography in Pleural Lesions",
      "diseases": [
        "Elasticity Imaging Techniques"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04792970",
      "trial_name": "Randomized Controlled Trial of Talc Instillation In Addition To Daily Drainage Through a Tunneled Pleural Catheter to Improve Rates of Outpatient Pleurodesis in Patients With Malignant Pleural Effusion",
      "diseases": [
        "Malignant Pleural Effusion"
      ],
      "ground_truth_label": 2
    },
    {
      "nct_id": "NCT04806373",
      "trial_name": "Intrapleural Fibrinolytic Therapy to Enhance Chemical Pleurodesis Enhance Chemical Pleurodesis",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 2
    }
  ],
  "partial_trials": [
    {
      "nct_id": "NCT00079235",
      "trial_name": "CCI-779 in Treating Patients With Stage IIIB (With Pleural Effusion) or Stage IV Non-Small Cell Lung Cancer",
      "diseases": [
        "Recurrent Non-small Cell Lung Cancer",
        "Stage IIIB Non-small Cell Lung Cancer",
        "Stage IV Non-small Cell Lung Cancer"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT00720954",
      "trial_name": "Rigid Thoracoscopy Versus CT-Guided Pleural Needle Biopsy",
      "diseases": [
        "Pleural Effusion",
        "Pleural Diseases"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT01062789",
      "trial_name": "Optical Breath-hold Control System for Image-guided Procedures",
      "diseases": [
        "Respiration"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT01831388",
      "trial_name": "Breath Training Exercise for the Reduction of Chronic Dyspnea",
      "diseases": [
        "Chronic Pulmonary Disorder"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT02505763",
      "trial_name": "Thoracic Ultrasound in the Treatment of Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT02807077",
      "trial_name": "PK of Pacritinib in Patients With Mild, Moderate, Severe Renal Impairment and ESRD Compared to Healthy Subjects",
      "diseases": [
        "Myelofibrosis"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT02834455",
      "trial_name": "Rational Approach to a Unilateral Pleural Effusion2",
      "diseases": [
        "Lung Neoplasms"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03272087",
      "trial_name": "Rational Approach to a Unilateral Pleural Effusion",
      "diseases": [
        "Lung Neoplasms"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03506295",
      "trial_name": "CrYobiopsy With Radial UltraSound Guidance",
      "diseases": [
        "Pulmonary Disease"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03550027",
      "trial_name": "Indwelling Pleural Catheter for Trapped Lung",
      "diseases": [
        "Pleura; Effusion"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03583931",
      "trial_name": "Treatment of Complicated Parapneumonic Effusion With Fibrinolytic Therapy Versus VATs Decortication",
      "diseases": [
        "Parapneumonic Effusion",
        "Empyema",
        "Pleural",
        "Coagulopathy"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03828903",
      "trial_name": "Role of Cryobiopsy in Diagnosis of Pleural Effusion",
      "diseases": [
        "Pleural Effusion"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03868579",
      "trial_name": "Rapid On Site Evaluation of Pleural Touch Preparations in Diagnosing Malignant Pleural Effusion in Patients Undergoing Pleuroscopy",
      "diseases": [
        "Malignant Neoplasm",
        "Malignant Respiratory Tract Neoplasm",
        "Malignant Thoracic Neoplasm"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT03881046",
      "trial_name": "Multidimensional Evaluation Of Daily Living Activities And Quality Of Life In Lung Cancer Patients",
      "diseases": [
        "Lung Cancer"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04233359",
      "trial_name": "A Randomised Study Evaluating Diagnostics of Pleural Effusion Among Patients Suspect of Cancer.",
      "diseases": [
        "Pleural Effusion",
        "Malignant",
        "Pleural Effusion",
        "Pleura; Exudate"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04235998",
      "trial_name": "Value of Additional Upfront Systematic Lung Ultrasound in the Workup of Patients With Unilateral Pleural Effusion",
      "diseases": [
        "Pleural Effusion",
        "Malignant Pleural Effusion",
        "Ultrasound"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04236037",
      "trial_name": "Ultrasound-guided Biopsy of the Pleura as a Supplement to Extraction of Fluid in Patients With One-sided Fluid in the Pleura",
      "diseases": [
        "Malignant Pleural Effusion",
        "Exudative Pleural Effusion"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04300244",
      "trial_name": "Nivolumab and Ipilimumab +/- UV1 Vaccination as Second Line Treatment in Patients With Malignant Mesothelioma",
      "diseases": [
        "Cancer",
        "Cancer",
        "Lung",
        "Cancer of Lung",
        "Mesothelioma",
        "Mesothelioma; Lung",
        "Mesothelioma; Pleura",
        "Mesotheliomas Pleural"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04553315",
      "trial_name": "the Effect of Chest Expansion Exercises on Pleural Effusion",
      "diseases": [
        "Pulmonary Infection",
        "Complication"
      ],
      "ground_truth_label": 1
    },
    {
      "nct_id": "NCT04749602",
      "trial_name": "Intrapleural Instillation of the Nivolumab in Cancer Patients With Pleural Effusion.",
      "diseases": [
        "Renal Cell Cancer Metastatic",
        "Non-small Cell Lung Cancer Metastatic",
        "Pleural Effusion",
        "Malignant"
      ],
      "ground_truth_label": 1
    }
  ],
  "excluded_trials_sample": [
    {
      "nct_id": "NCT00000644",
      "trial_name": "A Phase II Safety and Efficacy Study of Clarithromycin in the Treatment of Disseminated M. Avium Complex (MAC) Infections in Patients With AIDS",
      "diseases": [
        "Mycobacterium Avium-intracellulare Infection",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000672",
      "trial_name": "An Efficacy Study of 2',3'-Dideoxyinosine (ddI) (BMY-40900) Administered Orally Twice Daily to Zidovudine Intolerant Patients With AIDS or AIDS-Related Complex",
      "diseases": [
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000730",
      "trial_name": "Comparison of Three Treatments for Pneumocystis Pneumonia in AIDS Patients",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000739",
      "trial_name": "Comparison of Two Dosage Regimens of Oral Dapsone for Prophylaxis of Pneumocystis Carinii Pneumonia in Pediatric HIV Infection",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000768",
      "trial_name": "A Randomized Comparative Pharmacokinetic Study of Oral Ganciclovir After Treatment With Intravenous Ganciclovir for Cytomegalovirus Gastrointestinal Disease in AIDS Patients",
      "diseases": [
        "Colitis",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000769",
      "trial_name": "A Phase I/II Study of Recombinant Interleukin-4 in AIDS and Kaposi's Sarcoma",
      "diseases": [
        "Sarcoma",
        "Kaposi",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000773",
      "trial_name": "Phase I Safety and Pharmacokinetics Study of Microparticulate Atovaquone (m-Atovaquone; 566C80) in HIV-Infected and Perinatally Exposed Infants and Children",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000778",
      "trial_name": "A Pilot Study of Methodology to Rapidly Evaluate Drugs for Bactericidal Activity, Tolerance, and Pharmacokinetics in the Treatment of Pulmonary Tuberculosis Using Isoniazid and Levofloxacin",
      "diseases": [
        "HIV Infections",
        "Tuberculosis"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000816",
      "trial_name": "Gradual Initiation of Sulfamethoxazole/Trimethoprim as Primary Pneumocystis Carinii Pneumonia Prophylaxis",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000860",
      "trial_name": "The Effects of Treatment for Mycobacterium Avium Complex (MAC) on the Cells of HIV-Infected Patients",
      "diseases": [
        "Mycobacterium Avium-Intracellulare Infection",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000863",
      "trial_name": "A Study of WR 6026 in the Treatment of Pneumocystis Carinii Pneumonia (PCP) in HIV-Infected Patients",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00000970",
      "trial_name": "A Study of Foscarnet Plus Ganciclovir in the Treatment of Cytomegalovirus of the Eye in Patients With AIDS Who Have Already Been Treated With Ganciclovir",
      "diseases": [
        "Cytomegalovirus Retinitis",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001015",
      "trial_name": "A Study of Ribavirin in the Treatment of Patients With AIDS and AIDS-Related Problems",
      "diseases": [
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001030",
      "trial_name": "The Safety and Effectiveness of Clarithromycin and Rifabutin Used Alone or in Combination to Prevent Mycobacterium Avium Complex (MAC) or Disseminated MAC Disease in HIV-Infected Patients",
      "diseases": [
        "Mycobacterium Avium-intracellulare Infection",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001282",
      "trial_name": "A Phase I/II Study of the Combination of Azidothymidine and Interleukin-2 (IL-2) in the Treatment of HIV-Infected Patients",
      "diseases": [
        "HIV Infection"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001368",
      "trial_name": "Potential Risk Factors for Stroke",
      "diseases": [
        "Carotid Atherosclerosis",
        "Cerebrovascular Accident",
        "Diabetes Mellitus",
        "Hypercholesterolemia",
        "Hypertension"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001622",
      "trial_name": "Study of the Response of Human Small Blood Vessels",
      "diseases": [
        "Healthy",
        "Hypercholesterolemia",
        "Hypertension"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001742",
      "trial_name": "The Role of Cyclooxygenase Activity in the Endothelial Function of Hypertensive and Hypercholesterolemic Patients",
      "diseases": [
        "Healthy",
        "Hypercholesterolemia",
        "Hypertension"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001763",
      "trial_name": "Subcutaneously Administered Interleukin-12 Therapy in HIV-Infected Patients With Disseminated Mycobacterium Avium Complex Infection",
      "diseases": [
        "Acquired Immunodeficiency Syndrome",
        "Mycobacterium Avium-Intracellulare Infection"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001899",
      "trial_name": "Immune and Viral Status of HIV-Infected Patients After Stopping Combination Antiretroviral Therapy",
      "diseases": [
        "HIV Infection"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001911",
      "trial_name": "Interleukin-12 in the Treatment of Severe Nontuberculous Mycobacterial Infections",
      "diseases": [
        "Atypical Mycobacterium Infection"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001955",
      "trial_name": "Study of Etanercept and Celecoxib to Treat Temporomandibular Disorders (Painful Joint Conditions)",
      "diseases": [
        "Temporomandibular Joint Disorder"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001963",
      "trial_name": "Vascular Effects of Endothelium-Derived Versus Hemoglobin-Transported Nitric Oxide in Healthy Subjects",
      "diseases": [
        "Diabetes Mellitus",
        "Healthy",
        "Hypercholesterolemia",
        "Hypertension"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00001991",
      "trial_name": "A Treatment IND for 566C80 Therapy of Pneumocystis Carinii Pneumonia",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002032",
      "trial_name": "Rifabutin Therapy for the Prevention of Mycobacterium Avium Complex (MAC) Bacteremia in AIDS Patients With CD4 Counts = or < 200: A Double-Blind, Placebo-Controlled Trial",
      "diseases": [
        "Mycobacterium Avium-intracellulare Infection",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002103",
      "trial_name": "A Compassionate Treatment Protocol for the Use of Trimetrexate Glucuronate (Neutrexin) With Leucovorin Protection for European Adult Patients (>= 13 Years Old) With Pneumocystis Carinii Pneumonia",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002120",
      "trial_name": "Randomized Phase I Study of Trimetrexate Glucuronate (TMTX) With Leucovorin (LCV) Protection Plus Dapsone Versus Trimethoprim / Sulfamethoxazole (TMP/SMX) for Treatment of Moderately Severe Episodes of Pneumocystis Carinii Pneumonia",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002273",
      "trial_name": "A Study of Ganciclovir in the Treatment of Cytomegalovirus of the Large Intestine in Patients With AIDS",
      "diseases": [
        "Colitis",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002274",
      "trial_name": "A Study of ddI in Patients With AIDS Who Become Sicker While Taking Zidovudine",
      "diseases": [
        "HIV Infections",
        "Leukoencephalopathy",
        "Progressive Multifocal"
      ],
      "ground_truth_label": 0
    },
    {
      "nct_id": "NCT00002317",
      "trial_name": "A Study of Trimetrexate Plus Leucovorin in Children With Pneumocystis Carinii Pneumonia",
      "diseases": [
        "Pneumonia",
        "Pneumocystis Carinii",
        "HIV Infections"
      ],
      "ground_truth_label": 0
    }
  ],
  "excluded_trials_count": 509,
  "scoring_expectations": {
    "ideal_ranking": "All label=2 trials ranked above label=1, all label=1 above label=0",
    "notes": [
      "118 eligible trials cover mesothelioma treatment, pleural effusion management, and lung cancer diagnostics",
      "20 partially relevant trials address related but not perfectly matching conditions",
      "509 excluded trials target unrelated diseases or have criteria the patient cannot meet",
      "A well-performing system should achieve high NDCG by keeping mesothelioma and pleural effusion trials at the top"
    ]
  }
}


================================================
FILE: trial_matching_agent/data/trec2022_ground_truth/patient.jsonl
================================================
{"_id": "trec-20226", "source": "TREC 2022 Clinical Trials Track", "source_url": "https://github.com/ncbi-nlp/TrialGPT/tree/main/dataset/trec_2022", "disease_focus": "Malignant Pleural Mesothelioma / Lung Cancer", "text": "A 61-year-old man comes to the clinic due to nonproductive cough and progressive dyspnea.  The patient's medical conditions include hypertension, hypercholesteremia and peptic ulcer disease.  He smokes 2 packs of cigarettes daily for the past 30 years.  On examination, there are decreased breath sounds and percussive dullness at the base of the left lung.  Other vital signs are normal. Abdomen is soft without tenderness. CT scan shows a left-sided pleural effusion and nodular thickening of the pleura. The plural fluid was bloody on thoracentesis. Biopsy shows proliferation of epithelioid-type cells with very long microvilli.", "numbered_sentences": ["0. A 61-year-old man comes to the clinic due to nonproductive cough and progressive dyspnea.", "1. The patient's medical conditions include hypertension, hypercholesteremia and peptic ulcer disease.", "2. He smokes 2 packs of cigarettes daily for the past 30 years.", "3. On examination, there are decreased breath sounds and percussive dullness at the base of the left lung.", "4. Other vital signs are normal.", "5. Abdomen is soft without tenderness.", "6. CT scan shows a left-sided pleural effusion and nodular thickening of the pleura.", "7. The plural fluid was bloody on thoracentesis.", "8. Biopsy shows proliferation of epithelioid-type cells with very long microvilli."], "structured_profile": {"age": 61, "sex": "male", "diagnosis": "Malignant pleural mesothelioma (epithelioid type)", "diagnosis_evidence": "Biopsy shows proliferation of epithelioid-type cells with very long microvilli", "key_findings": {"CT_scan": "Left-sided pleural effusion and nodular thickening of the pleura", "thoracentesis": "Bloody pleural fluid", "biopsy": "Epithelioid-type cells with very long microvilli"}, "comorbidities": ["hypertension", "hypercholesterolemia", "peptic ulcer disease"], "smoking_history": "2 packs/day for 30 years (60 pack-years, current smoker)", "vital_signs": "normal", "physical_exam": {"lungs": "decreased breath sounds and percussive dullness at the base of the left lung", "abdomen": "soft without tenderness"}}, "gpt4_summary": "A 61-year-old man with a history of hypertension, hypercholesteremia, and peptic ulcer disease, who is a heavy smoker, presents with nonproductive cough, progressive dyspnea, decreased breath sounds, and percussive dullness at the base of the left lung. CT scan reveals a left-sided pleural effusion and nodular thickening of the pleura, with bloody pleural fluid and biopsy showing proliferation of epithelioid-type cells with very long microvilli."}



================================================
FILE: trial_matching_agent/data/trec2022_ground_truth/qrels.tsv
================================================
query-id	corpus-id	score
trec-20226	NCT00002622	2
trec-20226	NCT00002872	2
trec-20226	NCT00005636	2
trec-20226	NCT00019825	2
trec-20226	NCT00037817	2
trec-20226	NCT00042770	2
trec-20226	NCT00094978	2
trec-20226	NCT00114205	2
trec-20226	NCT00165555	2
trec-20226	NCT00166894	2
trec-20226	NCT00172835	2
trec-20226	NCT00227630	2
trec-20226	NCT00242723	2
trec-20226	NCT00313066	2
trec-20226	NCT00316134	2
trec-20226	NCT00430664	2
trec-20226	NCT00469196	2
trec-20226	NCT00473291	2
trec-20226	NCT00504738	2
trec-20226	NCT00509041	2
trec-20226	NCT00637078	2
trec-20226	NCT00637676	2
trec-20226	NCT00639067	2
trec-20226	NCT00644319	2
trec-20226	NCT00690261	2
trec-20226	NCT00715611	2
trec-20226	NCT00758316	2
trec-20226	NCT00789087	2
trec-20226	NCT00896285	2
trec-20226	NCT00978939	2
trec-20226	NCT01105390	2
trec-20226	NCT01125124	2
trec-20226	NCT01141946	2
trec-20226	NCT01193439	2
trec-20226	NCT01196585	2
trec-20226	NCT01262612	2
trec-20226	NCT01409551	2
trec-20226	NCT01411202	2
trec-20226	NCT01580982	2
trec-20226	NCT01601444	2
trec-20226	NCT01624090	2
trec-20226	NCT01756742	2
trec-20226	NCT01766739	2
trec-20226	NCT01778270	2
trec-20226	NCT01819363	2
trec-20226	NCT01952327	2
trec-20226	NCT01952431	2
trec-20226	NCT01973985	2
trec-20226	NCT02024113	2
trec-20226	NCT02045121	2
trec-20226	NCT02062632	2
trec-20226	NCT02092155	2
trec-20226	NCT02172027	2
trec-20226	NCT02192138	2
trec-20226	NCT02208895	2
trec-20226	NCT02227732	2
trec-20226	NCT02246946	2
trec-20226	NCT02269761	2
trec-20226	NCT02300883	2
trec-20226	NCT02320617	2
trec-20226	NCT02414269	2
trec-20226	NCT02436850	2
trec-20226	NCT02517749	2
trec-20226	NCT02518100	2
trec-20226	NCT02521922	2
trec-20226	NCT02524470	2
trec-20226	NCT02548221	2
trec-20226	NCT02583282	2
trec-20226	NCT02602119	2
trec-20226	NCT02613299	2
trec-20226	NCT02613312	2
trec-20226	NCT02627274	2
trec-20226	NCT02648763	2
trec-20226	NCT02649894	2
trec-20226	NCT02677883	2
trec-20226	NCT02724462	2
trec-20226	NCT02809872	2
trec-20226	NCT02825095	2
trec-20226	NCT02851927	2
trec-20226	NCT02975921	2
trec-20226	NCT03153501	2
trec-20226	NCT03172052	2
trec-20226	NCT03189108	2
trec-20226	NCT03270215	2
trec-20226	NCT03276715	2
trec-20226	NCT03290183	2
trec-20226	NCT03319186	2
trec-20226	NCT03319472	2
trec-20226	NCT03325192	2
trec-20226	NCT03403855	2
trec-20226	NCT03414905	2
trec-20226	NCT03482570	2
trec-20226	NCT03507452	2
trec-20226	NCT03591952	2
trec-20226	NCT03678090	2
trec-20226	NCT03736122	2
trec-20226	NCT03781908	2
trec-20226	NCT03831386	2
trec-20226	NCT03907852	2
trec-20226	NCT03922841	2
trec-20226	NCT03973957	2
trec-20226	NCT03997669	2
trec-20226	NCT04039126	2
trec-20226	NCT04074902	2
trec-20226	NCT04115111	2
trec-20226	NCT04264325	2
trec-20226	NCT04322136	2
trec-20226	NCT04413292	2
trec-20226	NCT04418804	2
trec-20226	NCT04420663	2
trec-20226	NCT04533854	2
trec-20226	NCT04670562	2
trec-20226	NCT04683419	2
trec-20226	NCT04731129	2
trec-20226	NCT04761003	2
trec-20226	NCT04781894	2
trec-20226	NCT04792970	2
trec-20226	NCT04806373	2
trec-20226	NCT00079235	1
trec-20226	NCT00720954	1
trec-20226	NCT01062789	1
trec-20226	NCT01831388	1
trec-20226	NCT02505763	1
trec-20226	NCT02807077	1
trec-20226	NCT02834455	1
trec-20226	NCT03272087	1
trec-20226	NCT03506295	1
trec-20226	NCT03550027	1
trec-20226	NCT03583931	1
trec-20226	NCT03828903	1
trec-20226	NCT03868579	1
trec-20226	NCT03881046	1
trec-20226	NCT04233359	1
trec-20226	NCT04235998	1
trec-20226	NCT04236037	1
trec-20226	NCT04300244	1
trec-20226	NCT04553315	1
trec-20226	NCT04749602	1
trec-20226	NCT00000644	0
trec-20226	NCT00000672	0
trec-20226	NCT00000730	0
trec-20226	NCT00000739	0
trec-20226	NCT00000768	0
trec-20226	NCT00000769	0
trec-20226	NCT00000773	0
trec-20226	NCT00000778	0
trec-20226	NCT00000816	0
trec-20226	NCT00000860	0
trec-20226	NCT00000863	0
trec-20226	NCT00000970	0
trec-20226	NCT00001015	0
trec-20226	NCT00001030	0
trec-20226	NCT00001282	0
trec-20226	NCT00001368	0
trec-20226	NCT00001622	0
trec-20226	NCT00001742	0
trec-20226	NCT00001763	0
trec-20226	NCT00001899	0
trec-20226	NCT00001911	0
trec-20226	NCT00001955	0
trec-20226	NCT00001963	0
trec-20226	NCT00001991	0
trec-20226	NCT00002032	0
trec-20226	NCT00002103	0
trec-20226	NCT00002120	0
trec-20226	NCT00002273	0
trec-20226	NCT00002274	0
trec-20226	NCT00002317	0
trec-20226	NCT00002343	0
trec-20226	NCT00002459	0
trec-20226	NCT00002550	0
trec-20226	NCT00002687	0
trec-20226	NCT00003364	0
trec-20226	NCT00003449	0
trec-20226	NCT00003623	0
trec-20226	NCT00003805	0
trec-20226	NCT00003938	0
trec-20226	NCT00003966	0
trec-20226	NCT00004883	0
trec-20226	NCT00004925	0
trec-20226	NCT00005293	0
trec-20226	NCT00005571	0
trec-20226	NCT00005610	0
trec-20226	NCT00005880	0
trec-20226	NCT00006417	0
trec-20226	NCT00020709	0
trec-20226	NCT00040339	0
trec-20226	NCT00040794	0
trec-20226	NCT00042835	0
trec-20226	NCT00057798	0
trec-20226	NCT00062231	0
trec-20226	NCT00062439	0
trec-20226	NCT00069303	0
trec-20226	NCT00094094	0
trec-20226	NCT00113386	0
trec-20226	NCT00167713	0
trec-20226	NCT00174369	0
trec-20226	NCT00175747	0
trec-20226	NCT00176098	0
trec-20226	NCT00188890	0
trec-20226	NCT00213603	0
trec-20226	NCT00215930	0
trec-20226	NCT00224198	0
trec-20226	NCT00226577	0
trec-20226	NCT00227617	0
trec-20226	NCT00241631	0
trec-20226	NCT00248495	0
trec-20226	NCT00250978	0
trec-20226	NCT00258518	0
trec-20226	NCT00258661	0
trec-20226	NCT00266877	0
trec-20226	NCT00268489	0
trec-20226	NCT00269399	0
trec-20226	NCT00274027	0
trec-20226	NCT00278148	0
trec-20226	NCT00298402	0
trec-20226	NCT00301249	0
trec-20226	NCT00301808	0
trec-20226	NCT00304356	0
trec-20226	NCT00304863	0
trec-20226	NCT00304889	0
trec-20226	NCT00307281	0
trec-20226	NCT00307346	0
trec-20226	NCT00311207	0
trec-20226	NCT00322751	0
trec-20226	NCT00329069	0
trec-20226	NCT00334815	0
trec-20226	NCT00335491	0
trec-20226	NCT00345774	0
trec-20226	NCT00350987	0
trec-20226	NCT00361972	0
trec-20226	NCT00378573	0
trec-20226	NCT00380315	0
trec-20226	NCT00391664	0
trec-20226	NCT00394147	0
trec-20226	NCT00395161	0
trec-20226	NCT00398775	0
trec-20226	NCT00402896	0
trec-20226	NCT00414960	0
trec-20226	NCT00428012	0
trec-20226	NCT00442468	0
trec-20226	NCT00450281	0
trec-20226	NCT00465907	0
trec-20226	NCT00471835	0
trec-20226	NCT00484783	0
trec-20226	NCT00502632	0
trec-20226	NCT00524147	0
trec-20226	NCT00534443	0
trec-20226	NCT00538018	0
trec-20226	NCT00540072	0
trec-20226	NCT00551369	0
trec-20226	NCT00556335	0
trec-20226	NCT00558636	0
trec-20226	NCT00560521	0
trec-20226	NCT00563329	0
trec-20226	NCT00564629	0
trec-20226	NCT00580892	0
trec-20226	NCT00608764	0
trec-20226	NCT00611650	0
trec-20226	NCT00614822	0
trec-20226	NCT00618813	0
trec-20226	NCT00624169	0
trec-20226	NCT00626938	0
trec-20226	NCT00629538	0
trec-20226	NCT00646490	0
trec-20226	NCT00665015	0
trec-20226	NCT00690885	0
trec-20226	NCT00691132	0
trec-20226	NCT00703339	0
trec-20226	NCT00706602	0
trec-20226	NCT00721487	0
trec-20226	NCT00723502	0
trec-20226	NCT00729456	0
trec-20226	NCT00750269	0
trec-20226	NCT00752076	0
trec-20226	NCT00753714	0
trec-20226	NCT00757120	0
trec-20226	NCT00762034	0
trec-20226	NCT00777569	0
trec-20226	NCT00798603	0
trec-20226	NCT00816504	0
trec-20226	NCT00826449	0
trec-20226	NCT00843726	0
trec-20226	NCT00850577	0
trec-20226	NCT00854139	0
trec-20226	NCT00855764	0
trec-20226	NCT00872612	0
trec-20226	NCT00880477	0
trec-20226	NCT00881595	0
trec-20226	NCT00883129	0
trec-20226	NCT00887315	0
trec-20226	NCT00888017	0
trec-20226	NCT00897143	0
trec-20226	NCT00925210	0
trec-20226	NCT00970801	0
trec-20226	NCT00973154	0
trec-20226	NCT00979212	0
trec-20226	NCT00981851	0
trec-20226	NCT00984243	0
trec-20226	NCT01004510	0
trec-20226	NCT01032694	0
trec-20226	NCT01050361	0
trec-20226	NCT01073722	0
trec-20226	NCT01080456	0
trec-20226	NCT01080469	0
trec-20226	NCT01080482	0
trec-20226	NCT01082107	0
trec-20226	NCT01085864	0
trec-20226	NCT01094431	0
trec-20226	NCT01110421	0
trec-20226	NCT01114386	0
trec-20226	NCT01114581	0
trec-20226	NCT01135979	0
trec-20226	NCT01139710	0
trec-20226	NCT01147562	0
trec-20226	NCT01157832	0
trec-20226	NCT01169038	0
trec-20226	NCT01170429	0
trec-20226	NCT01172925	0
trec-20226	NCT01183858	0
trec-20226	NCT01198587	0
trec-20226	NCT01225029	0
trec-20226	NCT01226992	0
trec-20226	NCT01230879	0
trec-20226	NCT01235182	0
trec-20226	NCT01245036	0
trec-20226	NCT01252199	0
trec-20226	NCT01257308	0
trec-20226	NCT01260116	0
trec-20226	NCT01261546	0
trec-20226	NCT01263132	0
trec-20226	NCT01263340	0
trec-20226	NCT01269892	0
trec-20226	NCT01276886	0
trec-20226	NCT01278017	0
trec-20226	NCT01280461	0
trec-20226	NCT01287429	0
trec-20226	NCT01306383	0
trec-20226	NCT01319266	0
trec-20226	NCT01325454	0
trec-20226	NCT01325753	0
trec-20226	NCT01371838	0
trec-20226	NCT01380769	0
trec-20226	NCT01385007	0
trec-20226	NCT01385930	0
trec-20226	NCT01387503	0
trec-20226	NCT01388192	0
trec-20226	NCT01405222	0
trec-20226	NCT01416961	0
trec-20226	NCT01420744	0
trec-20226	NCT01422239	0
trec-20226	NCT01424176	0
trec-20226	NCT01442779	0
trec-20226	NCT01448161	0
trec-20226	NCT01459770	0
trec-20226	NCT01467297	0
trec-20226	NCT01468909	0
trec-20226	NCT01504048	0
trec-20226	NCT01533012	0
trec-20226	NCT01561248	0
trec-20226	NCT01577862	0
trec-20226	NCT01591486	0
trec-20226	NCT01595178	0
trec-20226	NCT01612793	0
trec-20226	NCT01620645	0
trec-20226	NCT01621685	0
trec-20226	NCT01666548	0
trec-20226	NCT01667042	0
trec-20226	NCT01694654	0
trec-20226	NCT01724034	0
trec-20226	NCT01724671	0
trec-20226	NCT01731301	0
trec-20226	NCT01731821	0
trec-20226	NCT01742338	0
trec-20226	NCT01752647	0
trec-20226	NCT01773837	0
trec-20226	NCT01774526	0
trec-20226	NCT01776372	0
trec-20226	NCT01800188	0
trec-20226	NCT01812070	0
trec-20226	NCT01812278	0
trec-20226	NCT01835158	0
trec-20226	NCT01853644	0
trec-20226	NCT01863914	0
trec-20226	NCT01868373	0
trec-20226	NCT01871779	0
trec-20226	NCT01872585	0
trec-20226	NCT01874132	0
trec-20226	NCT01878175	0
trec-20226	NCT01879475	0
trec-20226	NCT01883869	0
trec-20226	NCT01902030	0
trec-20226	NCT01922180	0
trec-20226	NCT01926600	0
trec-20226	NCT01940328	0
trec-20226	NCT01944878	0
trec-20226	NCT02006940	0
trec-20226	NCT02009345	0
trec-20226	NCT02009787	0
trec-20226	NCT02024555	0
trec-20226	NCT02050022	0
trec-20226	NCT02054104	0
trec-20226	NCT02073968	0
trec-20226	NCT02084199	0
trec-20226	NCT02094950	0
trec-20226	NCT02099045	0
trec-20226	NCT02119494	0
trec-20226	NCT02119871	0
trec-20226	NCT02133781	0
trec-20226	NCT02137291	0
trec-20226	NCT02147821	0
trec-20226	NCT02150486	0
trec-20226	NCT02163954	0
trec-20226	NCT02168387	0
trec-20226	NCT02172300	0
trec-20226	NCT02182102	0
trec-20226	NCT02182232	0
trec-20226	NCT02198755	0
trec-20226	NCT02211898	0
trec-20226	NCT02216279	0
trec-20226	NCT02231047	0
trec-20226	NCT02269644	0
trec-20226	NCT02277366	0
trec-20226	NCT02304471	0
trec-20226	NCT02325024	0
trec-20226	NCT02341547	0
trec-20226	NCT02347722	0
trec-20226	NCT02347969	0
trec-20226	NCT02348398	0
trec-20226	NCT02360618	0
trec-20226	NCT02360865	0
trec-20226	NCT02363075	0
trec-20226	NCT02365168	0
trec-20226	NCT02365506	0
trec-20226	NCT02367222	0
trec-20226	NCT02391896	0
trec-20226	NCT02394548	0
trec-20226	NCT02434536	0
trec-20226	NCT02437760	0
trec-20226	NCT02480283	0
trec-20226	NCT02487394	0
trec-20226	NCT02500238	0
trec-20226	NCT02500693	0
trec-20226	NCT02538289	0
trec-20226	NCT02545036	0
trec-20226	NCT02559310	0
trec-20226	NCT02591550	0
trec-20226	NCT02595567	0
trec-20226	NCT02601053	0
trec-20226	NCT02608814	0
trec-20226	NCT02623166	0
trec-20226	NCT02638649	0
trec-20226	NCT02641600	0
trec-20226	NCT02644018	0
trec-20226	NCT02645968	0
trec-20226	NCT02662075	0
trec-20226	NCT02665546	0
trec-20226	NCT02674243	0
trec-20226	NCT02699385	0
trec-20226	NCT02703285	0
trec-20226	NCT02710968	0
trec-20226	NCT02716025	0
trec-20226	NCT02718066	0
trec-20226	NCT02719249	0
trec-20226	NCT02726451	0
trec-20226	NCT02771262	0
trec-20226	NCT02779478	0
trec-20226	NCT02782013	0
trec-20226	NCT02788890	0
trec-20226	NCT02808897	0
trec-20226	NCT02813694	0
trec-20226	NCT02827630	0
trec-20226	NCT02827734	0
trec-20226	NCT02850497	0
trec-20226	NCT02851407	0
trec-20226	NCT02855281	0
trec-20226	NCT02856217	0
trec-20226	NCT02856815	0
trec-20226	NCT02875782	0
trec-20226	NCT02882165	0
trec-20226	NCT02926534	0
trec-20226	NCT02928367	0
trec-20226	NCT02942043	0
trec-20226	NCT02946528	0
trec-20226	NCT02956213	0
trec-20226	NCT02968147	0
trec-20226	NCT02978144	0
trec-20226	NCT02985528	0
trec-20226	NCT03006679	0
trec-20226	NCT03026439	0
trec-20226	NCT03050034	0
trec-20226	NCT03053804	0
trec-20226	NCT03054454	0
trec-20226	NCT03096665	0
trec-20226	NCT03118882	0
trec-20226	NCT03139955	0
trec-20226	NCT03141216	0
trec-20226	NCT03143062	0
trec-20226	NCT03179072	0
trec-20226	NCT03179150	0
trec-20226	NCT03179267	0
trec-20226	NCT03191097	0
trec-20226	NCT03205995	0
trec-20226	NCT03206346	0
trec-20226	NCT03212248	0
trec-20226	NCT03213834	0
trec-20226	NCT03217409	0
trec-20226	NCT03226535	0
trec-20226	NCT03231072	0
trec-20226	NCT03240900	0
trec-20226	NCT03248713	0
trec-20226	NCT03249766	0
trec-20226	NCT03256396	0
trec-20226	NCT03259022	0
trec-20226	NCT03259087	0
trec-20226	NCT03260088	0
trec-20226	NCT03272997	0
trec-20226	NCT03275792	0
trec-20226	NCT03296787	0
trec-20226	NCT03310463	0
trec-20226	NCT03315403	0
trec-20226	NCT03320538	0
trec-20226	NCT03351335	0
trec-20226	NCT03359265	0
trec-20226	NCT03360331	0
trec-20226	NCT03362268	0
trec-20226	NCT03362281	0
trec-20226	NCT03362970	0
trec-20226	NCT03382106	0
trec-20226	NCT03408639	0
trec-20226	NCT03409055	0
trec-20226	NCT03423550	0
trec-20226	NCT03425617	0
trec-20226	NCT03432026	0
trec-20226	NCT03446261	0
trec-20226	NCT03446534	0
trec-20226	NCT03479983	0
trec-20226	NCT03480490	0
trec-20226	NCT03523845	0
trec-20226	NCT03526536	0
trec-20226	NCT03532750	0
trec-20226	NCT03545113	0
trec-20226	NCT03591562	0
trec-20226	NCT03606135	0
trec-20226	NCT03613779	0
trec-20226	NCT03617029	0
trec-20226	NCT03631472	0
trec-20226	NCT03660592	0
trec-20226	NCT03661801	0
trec-20226	NCT03691857	0
trec-20226	NCT03696524	0
trec-20226	NCT03773107	0
trec-20226	NCT03795662	0
trec-20226	NCT03798457	0
trec-20226	NCT03807050	0
trec-20226	NCT03811002	0
trec-20226	NCT03825770	0
trec-20226	NCT03837938	0
trec-20226	NCT03840603	0
trec-20226	NCT03852472	0
trec-20226	NCT03853551	0
trec-20226	NCT03853915	0
trec-20226	NCT03854136	0
trec-20226	NCT03854929	0
trec-20226	NCT03855670	0
trec-20226	NCT03856710	0
trec-20226	NCT03859206	0
trec-20226	NCT03861897	0
trec-20226	NCT03881657	0
trec-20226	NCT03883958	0
trec-20226	NCT03908944	0
trec-20226	NCT03910647	0
trec-20226	NCT03923673	0
trec-20226	NCT03944928	0
trec-20226	NCT03993899	0
trec-20226	NCT03998098	0
trec-20226	NCT04003415	0
trec-20226	NCT04003662	0
trec-20226	NCT04006405	0
trec-20226	NCT04012554	0
trec-20226	NCT04016181	0
trec-20226	NCT04022837	0
trec-20226	NCT04080232	0
trec-20226	NCT04088942	0
trec-20226	NCT04098185	0
trec-20226	NCT04124549	0
trec-20226	NCT04132375	0
trec-20226	NCT04142814	0
trec-20226	NCT04159831	0
trec-20226	NCT04177368	0
trec-20226	NCT04206098	0
trec-20226	NCT04224948	0
trec-20226	NCT04236349	0
trec-20226	NCT04236934	0
trec-20226	NCT04244604	0
trec-20226	NCT04279782	0
trec-20226	NCT04280523	0
trec-20226	NCT04289285	0
trec-20226	NCT04308278	0
trec-20226	NCT04311385	0
trec-20226	NCT04317690	0
trec-20226	NCT04326972	0
trec-20226	NCT04341558	0
trec-20226	NCT04355455	0
trec-20226	NCT04355676	0
trec-20226	NCT04363385	0
trec-20226	NCT04415255	0
trec-20226	NCT04432233	0
trec-20226	NCT04433039	0
trec-20226	NCT04444024	0
trec-20226	NCT04459286	0
trec-20226	NCT04481360	0
trec-20226	NCT04485650	0
trec-20226	NCT04492865	0
trec-20226	NCT04497311	0
trec-20226	NCT04498936	0
trec-20226	NCT04510077	0
trec-20226	NCT04532970	0
trec-20226	NCT04561219	0
trec-20226	NCT04569110	0
trec-20226	NCT04579367	0
trec-20226	NCT04584060	0
trec-20226	NCT04591093	0
trec-20226	NCT04627766	0
trec-20226	NCT04630301	0
trec-20226	NCT04677465	0
trec-20226	NCT04678050	0
trec-20226	NCT04681963	0
trec-20226	NCT04693312	0
trec-20226	NCT04695275	0
trec-20226	NCT04699656	0
trec-20226	NCT04724876	0
trec-20226	NCT04729842	0
trec-20226	NCT04733755	0
trec-20226	NCT04734236	0
trec-20226	NCT04749017	0
trec-20226	NCT04761289	0
trec-20226	NCT04763057	0
trec-20226	NCT04806412	0
trec-20226	NCT04808609	0
trec-20226	NCT04808752	0
trec-20226	NCT04815317	0
trec-20226	NCT04823741	0
trec-20226	NCT04845009	0
trec-20226	NCT04853199	0
trec-20226	NCT04860375	0
trec-20226	NCT04861649	0


