# Audit Table — medgemma-27b+gemini-flash-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-20144 | NCT00841789 | excl | Subject is currently enrolled in another investigational dev | MET | MET | MET | ✓ | The clinical note does not mention any current involvement in other trials or re |
| 2 | sigir-201422 | NCT00236912 | excl | Malnourished with low albumin | MET | MET | MET | ✓ | The clinical note does not mention malnutrition or low albumin levels. According |
| 3 | sigir-20146 | NCT00015626 | incl | Obese patients (wt > 95th percentile for age, for adults inc | MET | MET | MET | ✓ | The patient meets the age requirement (64 years old > 5 years) and is explicitly |
| 4 | sigir-20157 | NCT02633449 | excl | neurological diseases or relevant psychiatric diseases other | MET | MET | MET | ✓ | The patient note states there is no significant past medical history and only de |
| 5 | sigir-201422 | NCT00236912 | excl | HIV positive with current or past CD4 count < 200/mm^3 | MET | MET | MET | ✓ | The medical analysis confirms the clinical note does not mention the exclusion c |
| 6 | sigir-201410 | NCT02097186 | excl | Patients on glibenclamide or nicorandil (these medications m | MET | MET | MET | ✓ | The clinical note does not mention the use of glibenclamide or nicorandil. Accor |
| 7 | sigir-20144 | NCT02390596 | excl | Patients with any type of immunodeficiency or cancer | MET | MET | MET | ✓ | The medical analysis confirms that the patient does not have any history or find |
| 8 | sigir-20153 | NCT01326507 | excl | myocardial infarction in the 10 days before pulmonary emboli | MET | MET | MET | ✓ | The medical analysis confirms that the patient note does not mention any history |
| 9 | sigir-201425 | NCT00282269 | incl | are mechanically ventilated | NOT_MET | NOT_MET | UNKNOWN | ✗ | The medical analysis indicates that the patient note does not contain any inform |
| 10 | sigir-201523 | NCT02334514 | incl | clinical presentation that suggest influenza virus infection | NOT_MET | NOT_MET | MET | ✗ | The patient presents with several key symptoms listed in the criterion, includin |
| 11 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a non-smoker, but the medical |
| 12 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The clinical note confirms the patient is in respiratory distress, which is an e |
| 13 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The patient exhibits symptoms of cognitive impairment, specifically progressive  |
| 14 | sigir-20147 | NCT01863628 | excl | DSM-IV defined Axis I disorders | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient has a documented history of bipolar disorder, which is a DSM-IV defi |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing surgical lower limb revascularisation (s | NOT_MET | NOT_MET | UNKNOWN | ✗ | The clinical note describes symptoms of acute limb ischemia following a femoral  |
| 16 | sigir-201416 | NCT02238756 | excl | Acute disease at the time of enrolment. Acute disease is def | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient presented with severe acute symptoms including spastic movements, ag |
| 17 | sigir-201430 | NCT02273232 | incl | New claudication patient with Rutherford stage 2 and Fontain | UNKNOWN | UNKNOWN | MET | ✗ | The patient exhibits classic symptoms of intermittent claudication (calf pain on |
| 18 | sigir-201421 | NCT00997100 | excl | Female subject of child-bearing potential who is not using a | UNKNOWN | UNKNOWN | MET | ✗ | The clinical note identifies the patient as a 21-year-old female but does not me |
| 19 | sigir-201416 | NCT02374814 | excl | Subject is pregnant, or lactifying, or of childbearing poten | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient is identified as a 28-year-old female, which biologically implies ch |
| 20 | sigir-20141 | NCT01397994 | incl | Patients of chronic stable angina with abnormal Exercise Myo | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient's condition (new-onset chest pain) contradicts the inclusion require |

**Correct: 12/20 (60%)**
