# Audit Table — medgemma-27b+gemini-pro-two-stage-v4

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201427 | NCT01713881 | incl | Greater than 18 years old | MET | MET | MET | ✓ | The criterion requires the patient to be greater than 18 years old. The medical  |
| 2 | sigir-201415 | NCT00277680 | excl | Current or planned pregnancy | MET | MET | MET | ✓ | The criterion excludes patients with current or planned pregnancy. The medical a |
| 3 | sigir-20156 | NCT01717352 | excl | use of medications affecting sleep | MET | MET | MET | ✓ | The medical analysis indicates that while the patient has insomnia, the note doe |
| 4 | sigir-20144 | NCT00841789 | incl | Provision of Parental Consent | MET | MET | NOT_MET | ✗ | The criterion specifically requires 'Parental Consent' because the patient is a  |
| 5 | sigir-20145 | NCT00163709 | excl | Patients with severe renal disease (serum creatinine level o | MET | MET | MET | ✓ | The criterion is an exclusion for severe renal disease. The medical analysis rea |
| 6 | sigir-201519 | NCT00806091 | excl | none | MET | MET | NOT_MET | ✗ | The analysis confirms the patient has a 'chronic cough for the past two years',  |
| 7 | sigir-20147 | NCT00845988 | excl | history of neurological and medical illness | MET | MET | MET | ✓ | The medical analysis explicitly concludes that the patient does not have the exc |
| 8 | sigir-20151 | NCT00843063 | excl | psychosomatic disorder | MET | MET | MET | ✓ | The analysis concludes that the patient does not have the excluded condition (ps |
| 9 | sigir-20146 | NCT00015626 | excl | History of poor compliance with physician's recommendations | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis confirms the patient has the excluded condition ('not compliant wit |
| 10 | sigir-201413 | NCT02264769 | incl | Elective cesarean delivery under spinal anesthesia | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires an elective cesarean delivery under spinal anes |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with respiratory distress. The analysis confirms |
| 12 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The analysis concludes that the patient does not meet the criterion because the  |
| 13 | sigir-201522 | NCT02534727 | incl | Thought likely to be Mycobacterium culture positive (includi | NOT_MET | MET | NOT_MET | ✓ | The criterion requires the patient to be thought likely to be Mycobacterium cult |
| 14 | sigir-201521 | NCT01959048 | incl | Confirmed diagnosis of severe CDAD as defined above | NOT_MET | UNKNOWN | NOT_MET | ✓ | The analysis concludes that the patient does not have a confirmed diagnosis of s |
| 15 | sigir-201521 | NCT01959048 | excl | Other known etiology for diarrhea, or clinical infection wit | NOT_MET | NOT_MET | MET | ✗ | The analysis concludes that the patient does not have the excluded condition (An |
| 16 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis explicitly concludes 'NO' because the patient is being trea |
| 17 | sigir-201410 | NCT02097186 | excl | Pregnancy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |
| 18 | sigir-201516 | NCT00839124 | incl | Normal lung function, defined as (Knudson 1976/1984 predicte | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |
| 19 | sigir-20154 | NCT01243255 | incl | Patients on lipid lowering drug treatment | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |
| 20 | sigir-20147 | NCT00845988 | incl | patients who have exhibited a clinically significant increas | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |

**Correct: 17/20 (85%)**
