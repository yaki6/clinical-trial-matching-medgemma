# Audit Table — medgemma-27b+gemini-flash-two-stage-v3

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20157 | NCT01632319 | excl | If receiving pharmacological treatment for depression or sub | MET | MET | MET | ✓ | The medical analysis states that the patient is not receiving pharmacological tr |
| 2 | sigir-20153 | NCT01139632 | incl | Male and female 18years or older. | MET | MET | MET | ✓ | The patient is a 65-year-old male, which satisfies the inclusion requirement of  |
| 3 | sigir-201417 | NCT01745731 | excl | Severe heart failure (NYHA IV). | MET | MET | MET | ✓ | The medical analysis determined that the patient does not have the excluded cond |
| 4 | sigir-201524 | NCT00540072 | incl | Adults, 18 years of age or older of either gender and of any | MET | MET | NOT_MET | ✗ | The medical analysis confirms the patient meets the general inclusion criteria ( |
| 5 | sigir-201416 | NCT02238756 | excl | Administration of immunoglobulins (Igs) and/or any blood pro | MET | MET | MET | ✓ | The medical analysis found no evidence of the excluded treatment (immunoglobulin |
| 6 | sigir-201420 | NCT02255487 | incl | Is male or female, 18 years of age or older | MET | MET | MET | ✓ | The analysis confirms the patient is a 32-year-old woman, which satisfies both t |
| 7 | sigir-201421 | NCT00997100 | incl | Ability to sign and date a written informed consent prior to | MET | MET | MET | ✓ | The medical analysis confirms the patient can provide informed consent, which sa |
| 8 | sigir-201412 | NCT01551498 | excl | be taking any medication for treatment of autoimmune thyroid | MET | MET | UNKNOWN | ✗ | The medical analysis indicates there is insufficient data regarding the patient' |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The medical analysis concludes that the patient matches the inclusion criterion  |
| 10 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: The read operation timed out |
| 11 | sigir-201429 | NCT00703417 | incl | Postmenopausal female, 55-75 years old | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis determined that the patient does not meet the age requireme |
| 12 | sigir-201411 | NCT01526382 | incl | The use of vasopressors to maintain a systolic blood pressur | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis concluded 'NO' with high confidence, indicating the patient |
| 13 | sigir-201522 | NCT02534727 | incl | Thought likely to be Mycobacterium culture positive (includi | NOT_MET | MET | MET | ✗ | The medical analysis concludes that the patient matches the inclusion criterion  |
| 14 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | NOT_MET | ✓ | The medical analysis concludes with a definitive 'NO' for the inclusion criterio |
| 15 | sigir-201426 | NCT00373048 | excl | Intent to move out of the study catchment area during the st | NOT_MET | NOT_MET | UNKNOWN | ✗ | The medical analysis concluded that there was insufficient data to determine if  |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | NOT_MET | ✓ | The clinical analysis confirms that the patient presented with acute, severe sym |
| 17 | sigir-20149 | NCT01950026 | excl | suffering any change in skin sensitivity, illness of an infe | UNKNOWN | UNKNOWN | MET | ✗ | The medical analysis determined that the patient does not have any of the exclud |
| 18 | sigir-201521 | NCT01959048 | excl | Pregnant or lactating women | UNKNOWN | UNKNOWN | MET | ✗ | The clinical analysis confirms the patient is a 32-year-old male, which definiti |
| 19 | sigir-201513 | NCT01255670 | incl | patient speaks and understands Finnish or Swedish | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis indicates that there is insufficient data in the patient no |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate bone marrow function (WBC>3x10^9/l; Hb>9g/dl; plate | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis explicitly states that the patient note contains no informa |

**Correct: 12/20 (60%)**
