# BDD Conventions for trialmatch

## Directory Structure

```
features/                     # Gherkin feature files (specifications)
  {component}/                # One directory per src/trialmatch/ component
    {capability}.feature      # One file per capability

tests/bdd/                    # Step definitions (test code)
  conftest.py                 # Shared BDD fixtures
  steps/
    common_steps.py           # Reusable Given/When/Then steps
    {component}_steps.py      # Component-specific steps
```

Feature files live at project root (visible as specs). Step definitions live in `tests/bdd/` (test infrastructure).

## Tag Reference

| Tag | Purpose | Example |
|-----|---------|---------|
| `@wip` | Scenario written, not yet implemented | New feature file |
| `@implemented` | Step defs + source code passing | Green test |
| `@phase0` | Scoped to Phase 0 (20-pair eval) | Smoke tests |
| `@phase1` | Scoped to Phase 1 (Tier A/B eval) | Statistical tests |
| `@component_{name}` | Component isolation for selective runs | `@component_validate` |
| `@gold_input` | Uses gold SoT input (ADR-002 compliance) | Isolated component eval |
| `@needs_api` | Requires live API call (cost-aware) | Model integration tests |
| `@needs_data` | Requires TREC 2021 data on disk | Data loading tests |

## Tag Rules

1. Every scenario MUST have exactly one of `@wip` or `@implemented`
2. Every scenario MUST have a `@component_{name}` tag
3. Scenarios requiring API calls MUST have `@needs_api`
4. Scenarios using gold SoT input MUST have `@gold_input`

## File Naming

- Feature files: `features/{component}/{capability}.feature`
- Step files: `tests/bdd/steps/{component}_steps.py`
- Use snake_case for file names
- One capability per feature file (keep files focused)

## Step Definition Patterns

### Given steps (setup)
```python
@given("a TREC 2021 patient topic", target_fixture="patient")
def given_trec_patient(sample_trec_topic):
    return sample_trec_topic
```

### When steps (action)
```python
@when("the patient text is processed by INGEST", target_fixture="result")
def when_ingest_processes(patient, ingest_model):
    return ingest_model.understand(patient.text)
```

### Then steps (assertion)
```python
@then(parsers.parse("the result should contain at least {count:d} key facts"))
def then_keyfact_count(result, count):
    assert len(result.key_facts) >= count
```

### Shared fixtures (conftest.py)
```python
@pytest.fixture
def sample_trec_topic():
    """Load a known TREC 2021 topic for testing."""
    ...

@pytest.fixture
def mock_model_adapter():
    """Model adapter that returns canned responses (no API calls)."""
    ...
```

## Integration with Test Tiers

The same BDD feature exercises different depths depending on test tier:

| Tier | Model Backend | Data Source | Markers |
|------|--------------|-------------|---------|
| Unit (`tests/unit/`) | Mock/stub | Fixtures | `not needs_api` |
| Integration (`tests/integration/`) | VCR cassettes | Sample data | `integration` |
| E2E (`tests/e2e/`) | Live API | Real TREC data | `e2e and needs_api` |

BDD scenarios default to unit-level (mock backends). Add `@needs_api` for scenarios that must hit live APIs.

## Writing Good Scenarios

1. **Business language**: Describe WHAT, not HOW. "the patient profile should include demographics" not "the JSON should have a demographics key"
2. **One assertion per Then** (or closely related group)
3. **Background for shared setup**: Use `Background:` for Given steps common to all scenarios in a file
4. **Scenario Outline for parameterized tests**: Use when testing multiple inputs with same behavior
5. **Reference PRD sections**: Add comments linking scenarios to PRD requirement sections
