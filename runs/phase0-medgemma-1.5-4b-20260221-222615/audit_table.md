# Audit Table — medgemma-1.5-4b

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20144 | NCT00841789 | excl | Subject is currently enrolled in another investigational dev | MET | MET | MET | ✓ | The criterion requires the patient to be enrolled in another investigational dev |
| 2 | sigir-201422 | NCT00236912 | excl | Malnourished with low albumin | MET | MET | MET | ✓ | The criterion is 'Malnourished with low albumin'. The patient note does not ment |
| 3 | sigir-20146 | NCT00015626 | incl | Obese patients (wt > 95th percentile for age, for adults inc | MET | MET | NOT_MET | ✗ | The user wants me to determine the eligibility of a patient for a specific inclu |
| 4 | sigir-20157 | NCT02633449 | excl | neurological diseases or relevant psychiatric diseases other | MET | MET | MET | ✓ | The criterion is neurological diseases or relevant psychiatric diseases other th |
| 5 | sigir-201422 | NCT00236912 | excl | HIV positive with current or past CD4 count < 200/mm^3 | MET | MET | MET | ✓ | The criterion requires the patient to be HIV positive with a current or past CD4 |
| 6 | sigir-201410 | NCT02097186 | excl | Patients on glibenclamide or nicorandil (these medications m | MET | MET | MET | ✓ | The criterion requires the patient to be on glibenclamide or nicorandil. The pat |
| 7 | sigir-20144 | NCT02390596 | excl | Patients with any type of immunodeficiency or cancer | MET | MET | MET | ✓ | The criterion is 'Patients with any type of immunodeficiency or cancer'. The pat |
| 8 | sigir-20153 | NCT01326507 | excl | myocardial infarction in the 10 days before pulmonary emboli | MET | MET | MET | ✓ | The criterion requires the patient to have had a myocardial infarction within th |
| 9 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion is 'are mechanically ventilated'. The patient note does not mentio |
| 10 | sigir-201523 | NCT02334514 | incl | clinical presentation that suggest influenza virus infection | NOT_MET | NOT_MET | MET | ✗ | The criterion requires the patient to have a clinical presentation suggesting in |
| 11 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | MET | ✗ | The criterion is COPD, non smoker. The patient note explicitly states 'significa |
| 12 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The criterion is 'Respiratory distress'. The patient note states 'On examination |
| 13 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The criterion requires the patient to be experiencing symptoms of mild cognitive |
| 14 | sigir-20147 | NCT01863628 | excl | DSM-IV defined Axis I disorders | NOT_MET | NOT_MET | MET | ✗ | The criterion is DSM-IV defined Axis I disorders. The patient note mentions a hi |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing surgical lower limb revascularisation (s | NOT_MET | NOT_MET | MET | ✗ | The criterion requires patients undergoing surgical lower limb revascularisation |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires the patient to NOT have an acute disease at the time of e |
| 17 | sigir-201430 | NCT02273232 | incl | New claudication patient with Rutherford stage 2 and Fontain | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The user wants me to determine the eligibility of a patient for a specific inclu |
| 18 | sigir-201421 | NCT00997100 | excl | Female subject of child-bearing potential who is not using a | UNKNOWN | UNKNOWN | MET | ✗ | The criterion requires the patient to be female, of child-bearing potential, and |
| 19 | sigir-201416 | NCT02374814 | excl | Subject is pregnant, or lactifying, or of childbearing poten | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The criterion requires the patient to be pregnant, lactifying, or of childbearin |
| 20 | sigir-20141 | NCT01397994 | incl | Patients of chronic stable angina with abnormal Exercise Myo | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The user wants me to determine the eligibility of a patient for a specific inclu |

**Correct: 10/20 (50%)**
