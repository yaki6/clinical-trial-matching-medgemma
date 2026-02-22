# Audit Table — medgemma-4b+gemini-pro-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20144 | NCT00841789 | excl | Subject is currently enrolled in another investigational dev | MET | MET | MET | ✓ | The analysis states that the patient note does not explicitly mention enrollment |
| 2 | sigir-201422 | NCT00236912 | excl | Malnourished with low albumin | MET | MET | MET | ✓ | The criterion excludes patients who are 'Malnourished with low albumin'. The ana |
| 3 | sigir-20146 | NCT00015626 | incl | Obese patients (wt > 95th percentile for age, for adults inc | MET | MET | MET | ✓ | The criterion includes obese patients who are greater than 5 years of age. The p |
| 4 | sigir-20157 | NCT02633449 | excl | neurological diseases or relevant psychiatric diseases other | MET | MET | MET | ✓ | The criterion excludes patients with neurological or psychiatric diseases other  |
| 5 | sigir-201422 | NCT00236912 | excl | HIV positive with current or past CD4 count < 200/mm^3 | MET | MET | MET | ✓ | The medical analysis states that the patient note does not mention HIV status or |
| 6 | sigir-201410 | NCT02097186 | excl | Patients on glibenclamide or nicorandil (these medications m | MET | MET | MET | ✓ | The analysis determines that the patient does not have the excluded condition (t |
| 7 | sigir-20144 | NCT02390596 | excl | Patients with any type of immunodeficiency or cancer | MET | MET | MET | ✓ | The medical analysis indicates that the patient does not have immunodeficiency o |
| 8 | sigir-20153 | NCT01326507 | excl | myocardial infarction in the 10 days before pulmonary emboli | MET | MET | MET | ✓ | The analysis notes that the patient's record does not explicitly state they had  |
| 9 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be mechanically ventilated. The  |
| 10 | sigir-201523 | NCT02334514 | incl | clinical presentation that suggest influenza virus infection | NOT_MET | NOT_MET | MET | ✗ | The medical analysis concludes the patient satisfies the requirement for a clini |
| 11 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a non-smoker (specifically 'C |
| 12 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The analysis indicates the patient has the excluded condition, citing specific e |
| 13 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The criterion requires participants to be experiencing 'symptoms of mild cogniti |
| 14 | sigir-20147 | NCT01863628 | excl | DSM-IV defined Axis I disorders | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with DSM-IV defined Axis I disorders. The patien |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing surgical lower limb revascularisation (s | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires patients to be undergoing surgical lower limb revasculari |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | MET | ✗ | The medical analysis states that the patient note does not explicitly mention an |
| 17 | sigir-201430 | NCT02273232 | incl | New claudication patient with Rutherford stage 2 and Fontain | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The patient exhibits symptoms consistent with claudication (calf pain on exertio |
| 18 | sigir-201421 | NCT00997100 | excl | Female subject of child-bearing potential who is not using a | UNKNOWN | UNKNOWN | MET | ✗ | The criterion excludes female subjects of child-bearing potential who are not us |
| 19 | sigir-201416 | NCT02374814 | excl | Subject is pregnant, or lactifying, or of childbearing poten | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The provided medical analysis is incomplete (truncated) and does not contain fin |
| 20 | sigir-20141 | NCT01397994 | incl | Patients of chronic stable angina with abnormal Exercise Myo | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The analysis states that the patient note does not explicitly mention chronic st |

**Correct: 16/20 (80%)**
