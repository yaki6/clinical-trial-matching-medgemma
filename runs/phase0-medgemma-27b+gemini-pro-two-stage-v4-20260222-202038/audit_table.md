# Audit Table — medgemma-27b+gemini-pro-two-stage-v4

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20157 | NCT01632319 | excl | If receiving pharmacological treatment for depression or sub | MET | MET | MET | ✓ | The analysis concludes that the patient does not meet the exclusion criterion. T |
| 2 | sigir-20153 | NCT01139632 | incl | Male and female 18years or older. | MET | MET | MET | ✓ | The patient is a 65-year-old male, which satisfies the inclusion criteria of bei |
| 3 | sigir-201417 | NCT01745731 | excl | Severe heart failure (NYHA IV). | MET | MET | MET | ✓ | The criterion excludes patients with severe heart failure (NYHA IV). The medical |
| 4 | sigir-201524 | NCT00540072 | incl | Adults, 18 years of age or older of either gender and of any | MET | MET | UNKNOWN | ✗ | The patient meets the age and gender requirements (31-year-old male), but the cl |
| 5 | sigir-201416 | NCT02238756 | excl | Administration of immunoglobulins (Igs) and/or any blood pro | MET | MET | MET | ✓ | The analysis explicitly states that the patient note does not mention the admini |
| 6 | sigir-201420 | NCT02255487 | incl | Is male or female, 18 years of age or older | MET | MET | MET | ✓ | The patient is identified as a 32-year-old woman, which satisfies the inclusion  |
| 7 | sigir-201421 | NCT00997100 | incl | Ability to sign and date a written informed consent prior to | MET | MET | MET | ✓ | The medical analysis confirms that the patient note explicitly states the patien |
| 8 | sigir-201412 | NCT01551498 | excl | be taking any medication for treatment of autoimmune thyroid | MET | MET | MET | ✓ | The medical analysis indicates that the patient note does not mention any medica |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The criterion requires symptoms of 'mild' cognitive impairment. The patient's ne |
| 10 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | UNKNOWN | ✗ | The medical analysis returned a system error (timeout) instead of clinical findi |
| 11 | sigir-201429 | NCT00703417 | incl | Postmenopausal female, 55-75 years old | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient is 51 years old, which is outside the required age range of 55-75 ye |
| 12 | sigir-201411 | NCT01526382 | incl | The use of vasopressors to maintain a systolic blood pressur | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the use of vasopressors to maintain a systolic  |
| 13 | sigir-201522 | NCT02534727 | incl | Thought likely to be Mycobacterium culture positive (includi | NOT_MET | MET | NOT_MET | ✓ | The criterion requires the patient to be considered likely positive for Mycobact |
| 14 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis concludes that the patient does not satisfy the inclusion criterion |
| 15 | sigir-201426 | NCT00373048 | excl | Intent to move out of the study catchment area during the st | NOT_MET | NOT_MET | MET | ✗ | The analysis states that the patient note does not explicitly state whether the  |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis concludes the patient has an acute disease (symptoms including hydr |
| 17 | sigir-20149 | NCT01950026 | excl | suffering any change in skin sensitivity, illness of an infe | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |
| 18 | sigir-201521 | NCT01959048 | excl | Pregnant or lactating women | UNKNOWN | UNKNOWN | MET | ✗ | The patient is identified as a 32-year-old male, meaning he does not have the ex |
| 19 | sigir-201513 | NCT01255670 | incl | patient speaks and understands Finnish or Swedish | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis explicitly states that the patient note does not contain in |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate bone marrow function (WBC>3x10^9/l; Hb>9g/dl; plate | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded  |

**Correct: 16/20 (80%)**
