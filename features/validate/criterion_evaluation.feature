@component_validate @phase0
Feature: Criterion-level eligibility evaluation
  As a benchmark runner or e2e pipeline user
  I want to evaluate whether a patient meets a single eligibility criterion
  So that I can assess model medical reasoning quality

  @implemented
  Scenario: Prompt contains patient note and criterion
    Given a patient note "45-year-old male with NSCLC, ECOG 1"
    And an inclusion criterion "Histologically confirmed NSCLC"
    When I build the criterion evaluation prompt
    Then the prompt contains the patient note
    And the prompt contains the criterion text
    And the prompt asks for MET, NOT_MET, or UNKNOWN

  @implemented
  Scenario: Parse MET verdict from JSON
    Given the model returns '{"verdict": "MET", "reasoning": "confirmed NSCLC", "evidence_sentences": "0"}'
    When I parse the criterion verdict
    Then the verdict is MET
    And the reasoning contains "NSCLC"
    And evidence sentences include 0

  @implemented
  Scenario: Parse NOT_MET verdict from JSON
    Given the model returns '{"verdict": "NOT_MET", "reasoning": "too young", "evidence_sentences": ""}'
    When I parse the criterion verdict
    Then the verdict is NOT_MET
    And evidence sentences are empty

  @implemented
  Scenario: Evaluator is reusable (no benchmark coupling)
    Given the evaluate_criterion function
    Then it does not import from trialmatch.data
    And it does not import from trialmatch.cli
