# ADR-010: CT.gov Study Type Filtering via AREA[StudyType] Syntax

**Status:** Accepted
**Date:** 2026-02-22
**Decision Makers:** Claude + Yaqi

## Context

The CT.gov API v2 does **not** support `filter.studyType` as a query parameter, despite this being a common assumption. Attempting to use `filter.studyType=INTERVENTIONAL` returns a 400 Bad Request error or is silently ignored, meaning searches return both interventional and observational studies mixed together.

For clinical trial matching, returning observational studies alongside interventional trials confuses the agent and dilutes result quality — patients seeking enrollable treatment trials don't benefit from observational study results.

## Decision

Filter by study type using the CT.gov Essie advanced query syntax: `AREA[StudyType]Interventional` in the `query.term` parameter.

### Implementation

- **Location**: `src/trialmatch/prescreen/ctgov_client.py`
- **Mapping**: `_STUDY_TYPE_AREA_MAP` dict translates enum values to Essie values:
  - `INTERVENTIONAL` → `AREA[StudyType]Interventional`
  - `OBSERVATIONAL` → `AREA[StudyType]Observational`
  - `EXPANDED_ACCESS` → `AREA[StudyType]Expanded Access`
- **Composition**: AREA clauses are AND-joined with any existing `query.term` content (eligibility_keywords, age AREA expressions)
- **Default**: `INTERVENTIONAL` (set in `ToolExecutor._search_trials()`)
- **Tool schema**: `study_type` parameter added to `search_trials` tool declaration with enum validation

### Example API Call

```
GET /studies?query.cond=non-small+cell+lung+cancer
             &query.term=AREA[StudyType]Interventional AND AREA[MinimumAge]RANGE[MIN, 43 Years]
             &filter.overallStatus=RECRUITING
             &format=json
```

## Rationale

- CT.gov API v2 documents AREA[] as the Essie syntax for field-level queries. `StudyType` is a supported AREA field.
- The previous approach (`filter.studyType`) is not a valid API parameter — verified by 400 errors in production.
- AREA syntax integrates cleanly with the existing age range AREA expressions in `query.term`, using AND composition.
- Default to INTERVENTIONAL ensures the agent primarily finds enrollable clinical trials.

## Consequences

- **Pro:** Search results are filtered to the correct study type, improving result quality
- **Pro:** Consistent with how age bounds are already implemented (AREA syntax)
- **Pro:** Gemini agent can override to OBSERVATIONAL if clinically appropriate
- **Con:** Adds complexity to `query.term` composition (multiple AREA clauses AND-joined)
- **Con:** Combined `query.term` with keywords + AREA clauses may reduce result count vs. simpler queries

## Revisit When

- CT.gov API v2 adds a proper `filter.studyType` parameter
- AREA syntax behavior changes in future API versions
