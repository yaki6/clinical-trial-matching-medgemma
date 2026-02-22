# Audit Table — medgemma-4b+gemini-pro-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201427 | NCT01713881 | incl | Greater than 18 years old | MET | MET | MET | ✓ | The criterion requires the patient to be older than 18 years. The patient note i |
| 2 | sigir-201415 | NCT00277680 | excl | Current or planned pregnancy | MET | MET | NOT_MET | ✗ | The patient note describes a presentation of '12 weeks of amenorrhea' accompanie |
| 3 | sigir-20156 | NCT01717352 | excl | use of medications affecting sleep | MET | MET | MET | ✓ | The criterion excludes patients who use medications affecting sleep. The medical |
| 4 | sigir-20144 | NCT00841789 | incl | Provision of Parental Consent | MET | MET | MET | ✓ | The analysis indicates the criterion requires informed consent. The patient note |
| 5 | sigir-20145 | NCT00163709 | excl | Patients with severe renal disease (serum creatinine level o | MET | MET | MET | ✓ | The criterion excludes patients with severe renal disease. The medical analysis  |
| 6 | sigir-201519 | NCT00806091 | excl | none | MET | MET | MET | ✓ | The criterion is explicitly stated as "none", which implies there are no specifi |
| 7 | sigir-20147 | NCT00845988 | excl | history of neurological and medical illness | MET | MET | NOT_MET | ✗ | The criterion excludes patients with a history of neurological and medical illne |
| 8 | sigir-20151 | NCT00843063 | excl | psychosomatic disorder | MET | MET | MET | ✓ | The medical analysis indicates that the patient note does not mention psychosoma |
| 9 | sigir-20146 | NCT00015626 | excl | History of poor compliance with physician's recommendations | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with a history of poor compliance. The analysis  |
| 10 | sigir-201413 | NCT02264769 | incl | Elective cesarean delivery under spinal anesthesia | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis explicitly answers 'NO' to the question of whether the patient has  |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with respiratory distress. The patient note expl |
| 12 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The criterion explicitly requires a 'pulmonary nodule' identified on a 'CT' scan |
| 13 | sigir-201522 | NCT02534727 | incl | Thought likely to be Mycobacterium culture positive (includi | NOT_MET | MET | MET | ✗ | The medical analysis concludes that the patient meets the general condition desc |
| 14 | sigir-201521 | NCT01959048 | incl | Confirmed diagnosis of severe CDAD as defined above | NOT_MET | UNKNOWN | NOT_MET | ✓ | The inclusion criterion requires a confirmed diagnosis of severe CDAD (Clostridi |
| 15 | sigir-201521 | NCT01959048 | excl | Other known etiology for diarrhea, or clinical infection wit | NOT_MET | NOT_MET | MET | ✗ | The medical analysis indicates that the patient note does not explicitly state a |
| 16 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis explicitly answers 'No' to whether the patient has the general cond |
| 17 | sigir-201410 | NCT02097186 | excl | Pregnancy | UNKNOWN | UNKNOWN | MET | ✗ | The criterion excludes patients with pregnancy. The medical analysis states that |
| 18 | sigir-201516 | NCT00839124 | incl | Normal lung function, defined as (Knudson 1976/1984 predicte | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis states that the patient note does not provide any informati |
| 19 | sigir-20154 | NCT01243255 | incl | Patients on lipid lowering drug treatment | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The inclusion criterion requires the patient to be on lipid lowering drug treatm |
| 20 | sigir-20147 | NCT00845988 | incl | patients who have exhibited a clinically significant increas | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The clinical analysis indicates that the patient note does not explicitly state  |

**Correct: 14/20 (70%)**
