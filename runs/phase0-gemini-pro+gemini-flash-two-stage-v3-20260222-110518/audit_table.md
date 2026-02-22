# Audit Table — gemini-pro+gemini-flash-two-stage-v3

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201427 | NCT01713881 | incl | Greater than 18 years old | MET | MET | MET | ✓ | The clinical analysis confirms the patient is 21 years old, which satisfies the  |
| 2 | sigir-201415 | NCT00277680 | excl | Current or planned pregnancy | MET | MET | MET | ✓ | The medical analysis explicitly states that the excluded condition (current or p |
| 3 | sigir-20156 | NCT01717352 | excl | use of medications affecting sleep | MET | MET | UNKNOWN | ✗ | The medical analysis provides insufficient data regarding the patient's use of m |
| 4 | sigir-20144 | NCT00841789 | incl | Provision of Parental Consent | MET | MET | UNKNOWN | ✗ | The medical analysis confirms the patient will provide informed consent, but fin |
| 5 | sigir-20145 | NCT00163709 | excl | Patients with severe renal disease (serum creatinine level o | MET | MET | MET | ✓ | The medical analysis indicates that the exclusion condition (severe renal diseas |
| 6 | sigir-201519 | NCT00806091 | excl | none | MET | MET | MET | ✓ | The medical analysis indicates that no exclusion criteria were specified ('none' |
| 7 | sigir-20147 | NCT00845988 | excl | history of neurological and medical illness | MET | MET | NOT_MET | ✗ | The medical analysis indicates that the patient has a history of medical conditi |
| 8 | sigir-20151 | NCT00843063 | excl | psychosomatic disorder | MET | MET | MET | ✓ | The medical analysis confirms that the patient does not have the excluded condit |
| 9 | sigir-20146 | NCT00015626 | excl | History of poor compliance with physician's recommendations | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis indicates that the patient has a documented history of non- |
| 10 | sigir-201413 | NCT02264769 | incl | Elective cesarean delivery under spinal anesthesia | NOT_MET | NOT_MET | UNKNOWN | ✗ | The medical analysis indicates insufficient data to determine if the delivery wa |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis confirms the presence of respiratory distress (General Cond |
| 12 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The medical analysis indicates the patient does not have a pulmonary nodule on C |
| 13 | sigir-201522 | NCT02534727 | incl | Thought likely to be Mycobacterium culture positive (includi | NOT_MET | MET | NOT_MET | ✓ | The medical analysis concludes that the patient does not meet the requirement of |
| 14 | sigir-201521 | NCT01959048 | incl | Confirmed diagnosis of severe CDAD as defined above | NOT_MET | UNKNOWN | NOT_MET | ✓ | The medical analysis indicates that the patient does not have Clostridium diffic |
| 15 | sigir-201521 | NCT01959048 | excl | Other known etiology for diarrhea, or clinical infection wit | NOT_MET | NOT_MET | MET | ✗ | The medical analysis indicates that there is no evidence of other known etiologi |
| 16 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The clinical analysis indicates that the patient is undergoing cardiac catheteri |
| 17 | sigir-201410 | NCT02097186 | excl | Pregnancy | UNKNOWN | UNKNOWN | MET | ✗ | The medical analysis confirms that the exclusion condition (pregnancy) is not pr |
| 18 | sigir-201516 | NCT00839124 | incl | Normal lung function, defined as (Knudson 1976/1984 predicte | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis concludes that there is insufficient data because the clini |
| 19 | sigir-20154 | NCT01243255 | incl | Patients on lipid lowering drug treatment | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis determined there was insufficient data to confirm the patie |
| 20 | sigir-20147 | NCT00845988 | incl | patients who have exhibited a clinically significant increas | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis identifies insufficient data to determine if the patient is |

**Correct: 14/20 (70%)**
