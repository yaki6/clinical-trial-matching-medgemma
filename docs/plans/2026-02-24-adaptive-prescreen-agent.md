# Plan: Adaptive Agentic PRESCREEN — Gemini Pro + MedGemma-as-Tool

## Context

The current PRESCREEN agent uses a **rigid checklist-driven** architecture:
1. MedGemma 27B generates 12-15 condition terms (pre-search, one-shot)
2. Gemini Flash mechanically executes one `search_trials` per term
3. No adaptation based on intermediate results

**Results on TREC 2022** (mesothelioma patient, 118 eligible trials):
- MedGemma 27B guidance: **18.6% recall** (22/118)
- Gemini Pro fallback guidance: **20.3% recall** (24/118)
- MedGemma contributed **0 unique trials** vs Gemini Pro alone
- Ground truth analysis shows **84.7% recall is achievable** with 10+ diverse terms

**Root causes of low recall:**
1. **No adaptive reasoning**: Agent executes a flat list — doesn't learn from intermediate results
2. **MedGemma is wasted as pre-search**: One-shot guidance can't adapt to what CT.gov actually returns
3. **Vocabulary scatter**: 22 searches spread across near-duplicate terms instead of concentrating on high-yield ones (e.g., "pleural effusion" alone could reach 69.5%)
4. **No breadth-first strategy**: Specific terms searched first exhaust tool budget before broad terms

**Goal**: Replace rigid checklist with an **adaptive reasoning agent** that:
- Uses **Gemini 3 Pro** (not Flash) as the orchestrator — full reasoning capability
- Calls **MedGemma 27B as a tool** when it needs medical expertise
- **Dynamically decides** search terms, filters, and strategy based on intermediate results
- Implements **breadth-first → filter-narrowing** search pattern

---

## Architecture: Before vs After

### Before (Rigid Checklist)
```
MedGemma 27B ──(one-shot guidance)──► Gemini Flash ──(flat loop)──► CT.gov
                                           │
                                    Execute ALL 15 terms
                                    mechanically, no reasoning
```

### After (Adaptive Agent)
```
                    ┌─── consult_medical_expert ───► MedGemma 27B
                    │       (on-demand, 2-3 calls)
                    │
Gemini 3 Pro ──────┼─── search_trials ───► CT.gov API
  (reasoning        │       (dynamic terms + filters)
   agent)           │
                    └─── get_trial_details ───► CT.gov API
                            (5-10 promising trials)
```

**Key difference**: Gemini Pro **reasons about what to search next** based on what it's already found. MedGemma becomes a **callable expert** it consults when uncertain about medical terminology, not a one-shot oracle.

---

## Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `src/trialmatch/prescreen/agent.py` | New system prompt, new user template, remove pre-search MedGemma call, add Gemini Pro model param, update `run_prescreen_agent()` | Major rewrite of prompts + flow |
| `src/trialmatch/prescreen/tools.py` | Add `consult_medical_expert` tool declaration + executor method | Add ~60 lines |
| `scripts/run_trec_prescreen.py` | Wire Gemini Pro (not Flash), pass MedGemma adapter to ToolExecutor | ~20 lines |
| `scripts/run_trec_phase2.py` | Same wiring changes | ~20 lines |
| `tests/unit/test_prescreen.py` | New tests for `consult_medical_expert`, update existing tests | ~80 lines |

**NOT modified**: `ctgov_client.py` (already supports all needed parameters), `schema.py` (existing fields sufficient).

---

## Step 1: Add `consult_medical_expert` Tool (`tools.py`)

### Tool Declaration

```python
_CONSULT_EXPERT_DECL = genai_types.FunctionDeclaration(
    name="consult_medical_expert",
    description=(
        "Ask a specialized medical AI (MedGemma 27B) a clinical question. "
        "Use this when you need domain expertise that you're uncertain about:\n"
        "- What condition terms or synonyms to search for a diagnosis\n"
        "- Whether a specific biomarker is relevant to the patient's condition\n"
        "- What comorbidities or clinical presentations are associated with the diagnosis\n"
        "- Whether a trial's intervention is appropriate for the patient's treatment line\n"
        "- Medical terminology clarification for eligibility criteria interpretation\n\n"
        "COST: Each call takes ~8-15 seconds. Use judiciously — 2-3 calls per patient "
        "is typical. Do NOT call this for every search result or minor question."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["question"],
        properties={
            "question": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "The clinical question to ask. Be specific and include patient context. "
                    "Good: 'What condition terms should I search on CT.gov for a 61-year-old "
                    "male with epithelioid mesothelioma? Include related presentations and "
                    "broader categories.' "
                    "Bad: 'What is mesothelioma?'"
                ),
            ),
        },
    ),
)
```

### Executor Method (in `ToolExecutor`)

```python
async def _consult_medical_expert(self, question: str) -> tuple[dict, str]:
    """Call MedGemma 27B for clinical expertise."""
    if self._medgemma is None:
        return (
            {"error": "Medical expert not available. Proceed with your own clinical knowledge."},
            "MedGemma unavailable — skipped",
        )

    prompt = (
        "You are a clinical expert advising a clinical trial search agent.\n"
        "Answer concisely and actionably. Focus on specific medical terms, "
        "synonyms, and CT.gov condition vocabulary.\n\n"
        f"Question: {question}"
    )
    response = await self._medgemma.generate(prompt, max_tokens=1024)

    self.medgemma_calls += 1
    self.medgemma_cost += response.estimated_cost

    return (
        {"answer": response.text},
        f"MedGemma ({response.latency_ms:.0f}ms): {response.text[:150]}...",
    )
```

### Update `PRESCREEN_TOOLS`

```python
PRESCREEN_TOOLS = genai_types.Tool(
    function_declarations=[
        _SEARCH_TRIALS_DECL,
        _GET_DETAILS_DECL,
        _CONSULT_EXPERT_DECL,  # NEW
    ]
)
```

### Update `ToolExecutor.execute()`

Add dispatch:
```python
if tool_name == "consult_medical_expert":
    return await self._consult_medical_expert(**args)
```

---

## Step 2: Rewrite `PRESCREEN_SYSTEM_PROMPT` (`agent.py`)

Minimal prompt — give the goal, the tools, and the constraints. Let Gemini Pro reason about strategy itself.

```python
PRESCREEN_SYSTEM_PROMPT = """\
You are a clinical trial search agent. Your goal: find ALL potentially \
relevant trials on ClinicalTrials.gov for a given patient. \
Maximize recall — missing an eligible trial is worse than including \
an irrelevant one.

You have {max_tool_calls} tool calls. Use them wisely.

## Tools

### search_trials
Search ClinicalTrials.gov. Key parameters:
- **condition**: Disease or condition (e.g., "mesothelioma", "pleural effusion"). \
  This is the primary search lever. CT.gov indexes trials under many different \
  vocabulary terms — a trial for "pleural effusion" will NOT appear in a \
  "mesothelioma" search. Use diverse condition terms across multiple calls.
- **intervention**: Drug or therapy name (e.g., "pembrolizumab", "cisplatin").
- **eligibility_keywords**: Free-text searched inside eligibility criteria \
  (e.g., "EGFR L858R", "treatment naive", "prior platinum"). Best for \
  biomarkers and clinical phenotype terms that appear in inclusion/exclusion text.
- **status**: Recruitment status filter. Default: ["RECRUITING"]. \
  Set to ["RECRUITING", "NOT_YET_RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"] \
  for historical or comprehensive searches.
- **phase**, **location**, **study_type**: Optional narrowing filters.
- **page_size**: Results per call (default 50, max 100).
- **age/sex filters**: Available but NOT recommended — many trials lack \
  structured age data and the filter silently excludes them, hurting recall.

One condition term per call. Diverse terms across calls > repeated similar terms.

### get_trial_details
Fetch full eligibility criteria for a specific trial by NCT ID. \
Use when you need the inclusion/exclusion text to assess patient fit. \
Each call costs one tool use — prefer more searches over more detail fetches.

### consult_medical_expert
Ask MedGemma, a specialized medical AI, a clinical question. Use when you \
need domain expertise you're uncertain about — e.g., what related conditions \
share treatment approaches, what molecular features are clinically relevant, \
or what terminology CT.gov uses for a specific disease. \
Each call takes ~10 seconds. Typical usage: 1-3 calls per patient.\
"""
```

**Design philosophy**: Trust the frontier model's reasoning. The prompt provides:
1. **Goal**: maximize recall
2. **Budget**: `{max_tool_calls}` tool calls
3. **Clear tool usage guide**: what each parameter does, when to use it, what to avoid
4. **Key insight**: diversity of condition terms is the #1 lever (learned from all runs)
5. **Anti-pattern**: don't use age/sex filters (learned from data — 93% loss on some terms)
6. **MedGemma explained**: what it is, when to call it, expected latency
7. **Zero strategy prescription** — no phases, no ordering, no "search X first then Y"

---

## Step 3: Rewrite `PRESCREEN_USER_TEMPLATE` (`agent.py`)

Simplified — just the patient data, no instructions:

```python
PRESCREEN_USER_TEMPLATE = """\
## Patient Demographics
- Age: {age}
- Sex: {sex} (API value: {sex_api})

## Patient Profile

{patient_note}

## Extracted Key Facts

{key_facts_text}

Find ALL potentially relevant clinical trials for this patient.\
"""
```

**What changed**: Removed `{clinical_guidance_section}`, `{search_checklist}`, and all strategy instructions. The system prompt provides the goal; the user message is pure data.

---

## Step 4: Simplify `run_prescreen_agent()` Flow (`agent.py`)

### Remove pre-search MedGemma call

The entire `_get_clinical_guidance()` call (lines 327-375) is **removed from the main flow**. MedGemma is now called via `consult_medical_expert` tool inside the Gemini loop.

### New signature

```python
async def run_prescreen_agent(
    patient_note: str,
    key_facts: dict[str, Any],
    ingest_source: str,
    gemini_adapter: GeminiAdapter,
    medgemma_adapter: Any | None = None,  # Passed to ToolExecutor, not called directly
    max_tool_calls: int = 25,  # INCREASED from 20: budget for expert calls + searches
    trace_callback: TraceCallback | None = None,
    on_agent_text: Any | None = None,
) -> PresearchResult:
```

### Changes:
1. **Remove** lines 327-375 (pre-search MedGemma guidance generation)
2. **Remove** `_parse_clinical_guidance()` call and checklist building
3. **Remove** `allow_gemini_fallback` parameter (no longer needed)
4. **Pass** `medgemma_adapter` to `ToolExecutor(ctgov, medgemma=medgemma_adapter)`
5. **Inject** `max_tool_calls` into system prompt via `.format(max_tool_calls=max_tool_calls)`
6. **Increase** default `max_tool_calls` to 25 (budget: ~2 expert + ~15 search + ~3 detail + ~5 buffer)
7. **Keep** all existing: budget guard, error handling, retry logic, deduplication, scoring, tracing

### User message construction (simplified):

```python
age = str(key_facts.get("age", "unknown"))
sex_raw = str(key_facts.get("sex", key_facts.get("gender", "unknown")))
sex_api = _normalize_sex(sex_raw)
key_facts_text = _format_key_facts(key_facts)

user_message = PRESCREEN_USER_TEMPLATE.format(
    age=age,
    sex=sex_raw,
    sex_api=sex_api,
    patient_note=patient_note,
    key_facts_text=key_facts_text,
)
```

### Config change — use Gemini Pro model:

The caller (scripts) will pass a `GeminiAdapter` configured with `gemini-3-pro-preview` instead of `gemini-2.5-flash`. The agent.py code itself is model-agnostic — it uses whatever adapter is passed.

---

## Step 5: Update TREC Scripts

### `scripts/run_trec_prescreen.py`

```python
# Change: use Gemini Pro for the search agent
from trialmatch.models.gemini import GeminiAdapter

gemini_pro = GeminiAdapter(model="gemini-3-pro-preview")  # NOT flash

# Change: pass medgemma to ToolExecutor via run_prescreen_agent
result = await run_prescreen_agent(
    patient_note=patient_note,
    key_facts=key_facts,
    ingest_source="gold",
    gemini_adapter=gemini_pro,
    medgemma_adapter=medgemma_27b,  # Goes to ToolExecutor
    max_tool_calls=25,
)
```

### `scripts/run_trec_phase2.py`

Same changes. Comparison becomes:
- **Run A**: Gemini Pro + MedGemma-as-tool (new architecture)
- **Run B**: Gemini Pro only (no MedGemma, to isolate MedGemma's value)

---

## Step 6: Clean Up Dead Code (`agent.py`)

### Remove (no longer needed):
- `CLINICAL_REASONING_PROMPT` (lines 104-149) — MedGemma prompted via tool now
- `FALLBACK_SEARCH_CHECKLIST` (lines 151-159) — no more checklist
- `_parse_clinical_guidance()` (lines 162-206) — no more structured parsing
- `_build_search_checklist()` (lines 209-224) — no more checklist building
- `_get_clinical_guidance()` (lines 692-785) — pre-search call removed
- `_get_gemini_fallback_guidance()` (lines 227-275) — fallback removed
- `allow_gemini_fallback` parameter and related logic

### Keep:
- `_format_key_facts()`, `_normalize_sex()`, `_describe_query()`
- `_score_candidate()`, `_build_candidates()`
- `_generate_with_retry()`, `_emit_trace()`
- All budget guard and error handling logic
- `PresearchResult` schema fields (backward-compatible)

---

## Step 7: Update Unit Tests (`tests/unit/test_prescreen.py`)

### Remove tests for removed functions:
- `test_parse_clinical_guidance_*` (3 tests)
- `test_build_search_checklist` (1 test)
- `test_run_prescreen_no_adapter_no_fallback_raises` (1 test)
- `test_run_prescreen_no_adapter_with_fallback_continues` (1 test)

### New tests:
| Test | What it verifies |
|------|-----------------|
| `test_consult_medical_expert_returns_answer` | ToolExecutor calls MedGemma, returns answer dict |
| `test_consult_medical_expert_no_adapter_returns_error` | Graceful fallback when MedGemma unavailable |
| `test_consult_medical_expert_tracks_cost` | `medgemma_calls` and `medgemma_cost` incremented |
| `test_prescreen_tools_includes_expert` | PRESCREEN_TOOLS has 3 declarations |
| `test_agent_no_medgemma_still_works` | Agent completes without MedGemma (expert tool returns error) |

### Update existing tests:
- Remove `allow_gemini_fallback` from all test calls
- Tests that mock `medgemma_adapter` now pass it to `ToolExecutor` not agent directly
- Update mock tool call sequences to include potential `consult_medical_expert` calls

---

## Expected Behavior

We intentionally do NOT prescribe a specific trace. The point of using Gemini Pro
with a minimal prompt is to let the model decide its own strategy.

**What we expect to observe (success criteria):**
- Agent uses **diverse condition terms** — not just the primary diagnosis
- Agent calls `consult_medical_expert` **1-3 times** (not 0, not 10)
- Agent makes **15-20 search_trials calls** using the budget effectively
- Agent does NOT waste budget on `get_trial_details` for every result (3-5 max)
- Agent does NOT use age/sex filters (per system prompt constraint)
- Total unique NCTs returned: **300+** (vs current ~400 but with better yield)
- Recall on TREC eligible set: **>20%** (vs current 20.3% baseline)

**What we do NOT prescribe:**
- Search ordering (broad-first vs specific-first — let the model decide)
- When to consult MedGemma (upfront vs mid-search vs both)
- How many parallel calls per turn
- Which specific terms to search

**Anti-patterns to watch for (may need prompt adjustment):**
- Agent skips `consult_medical_expert` entirely → may need stronger nudge in prompt
- Agent calls `consult_medical_expert` >5 times → wasting budget on expert calls
- Agent repeats near-duplicate condition terms → diversity constraint not working
- Agent fetches details for >8 trials → wasting budget on details over searches
- Agent uses age/sex filters despite instruction → constraint not clear enough

---

## Verification

1. **Unit tests**: `uv run pytest tests/unit/test_prescreen.py -v` — all pass
2. **Tool declaration check**: Verify PRESCREEN_TOOLS has 3 function declarations
3. **Generalizability**: Read new prompts, confirm zero disease-specific terms
4. **TREC validation** (background):
   ```bash
   uv run python scripts/run_trec_phase2.py 2>&1 | tee /tmp/trec_adaptive_v1.log
   ```
   - Check tool call distribution: expert / search / detail calls
   - Check condition term diversity (unique terms searched)
   - Compare recall vs previous 20.3% baseline
   - Review agent reasoning in final summary
5. **Context window**: Check total tokens stays under 50K (Gemini Pro has 1M context)
6. **Cost**: Gemini Pro is ~10x more expensive per token than Flash — verify total cost stays reasonable (~$1-2 per patient)

---

## Implementation Order

```
Step 1 (tools.py)     — add consult_medical_expert tool     [15 min]
Step 2 (agent.py)     — rewrite PRESCREEN_SYSTEM_PROMPT      [10 min]
Step 3 (agent.py)     — rewrite PRESCREEN_USER_TEMPLATE       [5 min]
Step 4 (agent.py)     — simplify run_prescreen_agent()        [15 min]
Step 5 (scripts)      — update TREC scripts for Gemini Pro    [10 min]
Step 6 (agent.py)     — remove dead code                      [5 min]
Step 7 (tests)        — new + updated tests                   [15 min]
```

Steps 1 and 2-3 are independent (tool vs prompt).
Step 4 depends on Steps 1-3.
Steps 5-7 depend on Step 4.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Gemini Pro costs ~10x more than Flash | Budget guard at 25 calls; monitor cost in trace |
| MedGemma endpoint down | `consult_medical_expert` returns graceful error; agent proceeds with own knowledge |
| Gemini Pro makes fewer parallel calls | System prompt encourages parallel calls; budget increased to 25 |
| Agent over-uses consult_medical_expert | Tool description says "2-3 calls typical"; system prompt limits to Phase 1 + Phase 4 |
| Loss of reproducibility | Keep trace_callback for full audit trail |

---

## IMPORTANT: Undeploy Vertex 27B After Validation

The Vertex 27B endpoint (`7499477442478735360`) is currently deployed at ~$2.30/hr. After TREC validation completes, undeploy immediately:
```bash
uv run python scripts/deploy_vertex_27b.py undeploy
```
