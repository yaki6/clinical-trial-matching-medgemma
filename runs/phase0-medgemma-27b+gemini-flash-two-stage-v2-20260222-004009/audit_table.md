# Audit Table — medgemma-27b+gemini-flash-two-stage-v2

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20144 | NCT00841789 | excl | Subject is currently enrolled in another investigational dev | MET | MET | MET | ✓ | The clinical note does not mention enrollment in other trials or the use of inve |
| 2 | sigir-201422 | NCT00236912 | excl | Malnourished with low albumin | MET | MET | MET | ✓ | The patient note contains no mention of malnutrition or low albumin levels; for  |
| 3 | sigir-20146 | NCT00015626 | incl | Obese patients (wt > 95th percentile for age, for adults inc | MET | MET | MET | ✓ | The patient is 64 years old, which exceeds the age requirement of 5 years, and i |
| 4 | sigir-20157 | NCT02633449 | excl | neurological diseases or relevant psychiatric diseases other | MET | MET | MET | ✓ | The patient has no significant past medical history and presents symptoms consis |
| 5 | sigir-201422 | NCT00236912 | excl | HIV positive with current or past CD4 count < 200/mm^3 | MET | MET | MET | ✓ | The clinical note does not mention HIV or CD4 counts. For an exclusion criterion |
| 6 | sigir-201410 | NCT02097186 | excl | Patients on glibenclamide or nicorandil (these medications m | MET | MET | MET | ✓ | The patient note does not mention the use of glibenclamide or nicorandil. Per th |
| 7 | sigir-20144 | NCT02390596 | excl | Patients with any type of immunodeficiency or cancer | MET | MET | MET | ✓ | The medical analysis indicates that the patient does not have the excluded condi |
| 8 | sigir-20153 | NCT01326507 | excl | myocardial infarction in the 10 days before pulmonary emboli | MET | MET | MET | ✓ | The clinical analysis confirms that the patient has no history of cardiovascular |
| 9 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | NOT_MET | ✓ | {   "label": "not_eligible",   "reasoning": "The medical analysis explicitly con |
| 10 | sigir-201523 | NCT02334514 | incl | clinical presentation that suggest influenza virus infection | NOT_MET | NOT_MET | MET | ✗ | The patient exhibits several key symptoms specified in the criterion for a clini |
| 11 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a non-smoker with COPD. Altho |
| 12 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient is explicitly stated to be in respiratory distress, which is an excl |
| 13 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The patient exhibits progressive memory loss and cognitive deficits, which are s |
| 14 | sigir-20147 | NCT01863628 | excl | DSM-IV defined Axis I disorders | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient has a documented history of bipolar disorder, which is classified as |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing surgical lower limb revascularisation (s | NOT_MET | NOT_MET | UNKNOWN | ✗ | The medical analysis states that while the patient has symptoms suggestive of a  |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient presented with severe acute symptoms including spasticity, agitation |
| 17 | sigir-201430 | NCT02273232 | incl | New claudication patient with Rutherford stage 2 and Fontain | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The patient presents with symptoms of new intermittent claudication; however, th |
| 18 | sigir-201421 | NCT00997100 | excl | Female subject of child-bearing potential who is not using a | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis reveals that there is no information in the patient's note  |
| 19 | sigir-201416 | NCT02374814 | excl | Subject is pregnant, or lactifying, or of childbearing poten | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient is a 28-year-old female, which the medical analysis identifies as ha |
| 20 | sigir-20141 | NCT01397994 | incl | Patients of chronic stable angina with abnormal Exercise Myo | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient presents with new-onset chest pain, which is inconsistent with the ' |

**Correct: 15/20 (75%)**
