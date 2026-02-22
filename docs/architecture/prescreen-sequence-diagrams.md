# Sequence Diagrams: PRESCREEN & Pipeline Interactions

> Generated from source: `src/trialmatch/prescreen/agent.py`, `tools.py`, `ctgov_client.py`, `cli/phase0.py`
>
> **Updated 2026-02-22**: Reflects current implementation — MedGemma clinical reasoning
> pre-search step (ADR-009), two tools only (normalize_medical_terms commented out — ADR-011),
> AREA[StudyType] filtering (ADR-010), heuristic candidate scoring.

---

## 1. PRESCREEN Agent Loop

The PRESCREEN pipeline has two phases:
1. **MedGemma clinical reasoning** (optional): MedGemma 4B generates clinical
   guidance (condition terms, molecular drivers, eligibility keywords) that is
   injected into Gemini's prompt.
2. **Gemini agentic loop**: Gemini autonomously searches ClinicalTrials.gov
   using two tools: `search_trials` and `get_trial_details`. The loop runs
   up to `max_tool_calls + 5` iterations, with a budget guard that sends
   structured FunctionResponse errors when the tool call cap is reached.

Note: `normalize_medical_terms` was the original third tool but was commented
out (ADR-011) due to ~25s latency with near-zero value.

```mermaid
sequenceDiagram
    participant CLI
    participant Agent as Agent<br/>(run_prescreen_agent)
    participant MG as MedGemma 4B
    participant Gemini as Gemini Flash
    participant TE as ToolExecutor
    participant CTGov as CTGovClient
    participant API as CT.gov API v2

    CLI->>Agent: run_prescreen_agent(patient_note, key_facts, ...)
    activate Agent

    Agent->>Agent: Create CTGovClient + ToolExecutor
    Agent->>Agent: Extract demographics (age, sex) from key_facts

    opt MedGemma clinical reasoning (if medgemma_adapter provided)
        Agent->>MG: generate(CLINICAL_REASONING_PROMPT)
        activate MG
        MG-->>Agent: Clinical guidance (condition terms, molecular drivers, keywords)
        deactivate MG
        Agent->>Agent: Inject guidance into user prompt as "MedGemma Clinical Reasoning" section
    end

    Agent->>Agent: Format user message via PRESCREEN_USER_TEMPLATE (with demographics + guidance)
    Agent->>Agent: Build contents[] with system prompt + user message

    loop Agentic Loop (max_tool_calls + 5 iterations)
        Agent->>Gemini: generate_content(contents with tool results)
        activate Gemini
        Gemini-->>Agent: response with function_calls[] and/or text
        deactivate Gemini

        alt No function_calls in response
            Note over Agent: Gemini finished — break loop
        else call_index >= max_tool_calls (Budget Guard)
            Agent->>Agent: Build FunctionResponse with error for each pending FC
            Note over Agent: "Tool budget of N calls exhausted.<br/>Stop searching and summarise findings."
            Agent->>Agent: Append error FunctionResponses to contents[]
            Note over Agent: continue loop (Gemini will produce final summary)
        else Normal tool execution
            Agent->>TE: execute("search_trials", {condition, intervention, ...})
            activate TE
            TE->>CTGov: search(condition, intervention, eligibility_keywords, ...)
            activate CTGov
            CTGov->>API: GET /studies?query.cond=...&aggFilters=...
            activate API
            API-->>CTGov: {studies[], totalCount}
            deactivate API
            CTGov-->>TE: raw response dict
            deactivate CTGov
            TE->>TE: parse_search_results() → parse_study_summary() → compact dicts
            TE-->>Agent: ({count, total_available, trials[]}, summary)
            deactivate TE

            Agent->>Agent: Merge trials into candidates_by_nct, track found_by_query

            opt get_trial_details (only for promising trials)
                Agent->>TE: execute("get_trial_details", {nct_id})
                activate TE
                TE->>CTGov: get_details(nct_id)
                activate CTGov
                CTGov->>API: GET /studies/{nct_id}
                activate API
                API-->>CTGov: full study record
                deactivate API
                CTGov-->>TE: raw response dict
                deactivate CTGov
                TE->>TE: Extract eligibility criteria, age, sex
                TE-->>Agent: ({nct_id, eligibility_criteria, ...}, summary)
                deactivate TE
                Agent->>Agent: Enrich candidate in candidates_by_nct
            end

            Agent->>Agent: Record ToolCallRecord per call, increment call_index
            Agent->>Agent: Append all FunctionResponse parts to contents[]
        end
    end

    Agent->>Agent: _build_candidates() — heuristic score: query_count*3 + phase_bonus + recruiting_bonus + details_bonus
    Agent->>Agent: Prune to MAX_CANDIDATES=20
    Agent->>Agent: Compute gemini_cost from token counts
    Agent->>CTGov: aclose()

    Agent-->>CLI: PresearchResult{candidates[], agent_reasoning, tool_call_trace, costs}
    deactivate Agent
```

---

## 2. CT.gov Parameter Mapping (Single search_trials Call)

Shows how Gemini's high-level search parameters are mapped to
ClinicalTrials.gov API v2 HTTP query parameters. Key details:
- `phase` and `sex` are joined into `aggFilters` (comma-separated)
- `min_age`/`max_age` become Essie `AREA[]RANGE[]` expressions in `query.term`
- `study_type` becomes `AREA[StudyType]Interventional` in `query.term` (ADR-010)
- `eligibility_keywords`, age, and study_type AREA clauses are AND-composed in `query.term`
- `filter.studyType` is NOT a valid API parameter — must use AREA syntax

```mermaid
sequenceDiagram
    participant Gemini as Gemini 3 Pro
    participant TE as ToolExecutor
    participant Client as CTGovClient
    participant API as CT.gov API v2

    Gemini->>TE: search_trials({condition, intervention,<br/>eligibility_keywords, phase, sex,<br/>min_age, max_age, location, status, page_size})
    activate TE

    TE->>TE: Cap page_size at 100
    TE->>TE: Default status to ["RECRUITING"] if not provided

    TE->>Client: search(condition, intervention, eligibility_keywords,<br/>status, phase, location, sex, min_age, max_age, page_size)
    activate Client

    Note over Client: Build HTTP params dict

    Note over Client: query.cond = condition<br/>query.intr = intervention<br/>query.locn = location

    Note over Client: Build age Essie clauses:<br/>min_age → AREA[MinimumAge]RANGE[MIN, {min_age}]<br/>max_age → AREA[MaximumAge]RANGE[{max_age}, MAX]<br/>AND-join age clauses

    Note over Client: Compose query.term:<br/>eligibility_keywords AND (age_essie_clauses)<br/>If no keywords: just age_essie_clauses

    Note over Client: Build aggFilters (comma-joined):<br/>phase: PHASE1→phase:1, PHASE2→phase:2, ...<br/>sex: MALE→sex:m, FEMALE→sex:f<br/>Example: "phase:2,phase:3,sex:f"

    Note over Client: filter.overallStatus = comma-joined status[]<br/>pageSize = page_size (capped 100)

    Client->>Client: Rate limit: enforce _MIN_INTERVAL (1.5s)

    Client->>API: GET /studies?query.cond={cond}<br/>&query.intr={intr}&query.locn={locn}<br/>&query.term={keywords AND age_essie}<br/>&aggFilters={phase,sex}<br/>&filter.overallStatus={status}<br/>&pageSize={n}&format=json
    activate API

    alt HTTP 200 OK
        API-->>Client: {studies: [...], totalCount: N}
    else HTTP 429 (Rate Limited)
        API-->>Client: 429 Too Many Requests
        Client->>Client: Exponential backoff: sleep(2^attempt)
        Client->>Client: Reset _last_call_time (prevent double-wait)
        Client->>API: Retry GET /studies?...
        API-->>Client: {studies: [...], totalCount: N}
    else HTTP 400 (Bad Request)
        API-->>Client: 400 + error body
        Client-->>TE: Raise ValueError("CT.gov API 400 Bad Request")
    end
    deactivate API

    Client-->>TE: raw response dict
    deactivate Client

    TE->>TE: parse_search_results(raw) → studies[]
    TE->>TE: For each study: parse_study_summary()<br/>Extract: nct_id, brief_title, phase,<br/>conditions, interventions, status,<br/>sponsor, enrollment

    TE->>TE: Build compact dicts for Gemini<br/>(conditions[:3], interventions[:4])

    TE-->>Gemini: {count, total_available, trials: [compact...]}
    deactivate TE
```

---

## 3. Phase 0 Benchmark

The benchmark harness that evaluates criterion-level matching.
Loads TrialGPT HF annotations, samples pairs, runs each model,
computes metrics, and persists run artifacts.

```mermaid
sequenceDiagram
    participant CLI as CLI (phase0_cmd)
    participant P0 as run_phase0()
    participant HF as HFLoader
    participant Samp as Sampler
    participant Eval as run_model_benchmark()
    participant VE as evaluate_criterion()
    participant Model as Model Adapter<br/>(MedGemma / Gemini)
    participant Metrics as compute_metrics()
    participant RM as RunManager

    CLI->>P0: phase0_cmd(config_path, dry_run)
    activate P0

    P0->>P0: Load + parse YAML config

    P0->>HF: load_annotations() or load_annotations_from_file()
    activate HF
    HF-->>P0: CriterionAnnotation[] (1,024 pairs)
    deactivate HF

    opt keyword_filter in config
        P0->>P0: filter_by_keywords(annotations, keywords)
    end

    P0->>Samp: stratified_sample(annotations, n_pairs=20, seed=42)
    activate Samp
    Samp-->>P0: Phase0Sample with sampled pairs
    deactivate Samp

    loop For each model in config.models[]
        P0->>P0: Instantiate adapter (MedGemmaAdapter or GeminiAdapter)

        P0->>Eval: run_model_benchmark(adapter, sample, budget_max, max_concurrent)
        activate Eval

        loop For each pair in sample.pairs (sequential or concurrent)
            Eval->>VE: evaluate_criterion(note, criterion_text, criterion_type, adapter)
            activate VE
            VE->>Model: generate(prompt with patient note + criterion)
            activate Model
            Model-->>VE: ModelResponse{text, verdict, tokens, cost, latency}
            deactivate Model
            VE-->>Eval: CriterionResult{verdict, evidence_sentences, model_response}
            deactivate VE
            Eval->>Eval: Accumulate cost, check budget guard
        end

        Eval-->>P0: CriterionResult[]
        deactivate Eval

        P0->>Metrics: compute_metrics(predicted_verdicts, expert_labels)
        activate Metrics
        Metrics-->>P0: {accuracy, f1_macro, f1_met_not_met, cohens_kappa, confusion_matrix}
        deactivate Metrics

        P0->>P0: compute_evidence_overlap() per pair
        P0->>P0: compute_metrics(gpt4_labels, expert_labels) — baseline
        P0->>P0: aggregate_to_trial_verdict() per (patient, trial) group

        P0->>RM: generate_run_id(model_name)
        activate RM
        P0->>RM: save_run(RunResult{run_id, results, metrics}, config, annotations)
        RM->>RM: Write to runs/{run_id}/<br/>config.json, results.json,<br/>metrics.json, cost_summary.json
        RM-->>P0: run_dir path
        deactivate RM

        P0->>CLI: Echo results table (accuracy, F1, kappa, GPT-4 baseline)
    end

    deactivate P0
```

---

## 4. Future E2E Pipeline

The planned end-to-end flow connecting all three components:
INGEST extracts the patient profile, PRESCREEN finds candidate trials,
and VALIDATE evaluates each eligibility criterion. Error propagation
is tested here (unlike isolated component evaluation which uses gold SoT inputs).

```mermaid
sequenceDiagram
    participant CLI
    participant Ingest as INGEST
    participant PS as PRESCREEN Agent
    participant CTGov as CT.gov API v2
    participant Val as VALIDATE
    participant Agg as Aggregator

    CLI->>Ingest: understand(patient_note)
    activate Ingest
    Ingest-->>CLI: PatientProfile + KeyFacts
    deactivate Ingest

    CLI->>PS: run_prescreen_agent(patient_note, key_facts, ...)
    activate PS

    loop Agentic search (Gemini-driven)
        PS->>CTGov: search_trials(condition, intervention, ...)
        CTGov-->>PS: trial summaries[]
    end

    opt Promising candidates
        PS->>CTGov: get_trial_details(nct_id)
        CTGov-->>PS: full eligibility criteria text
    end

    PS-->>CLI: PresearchResult{TrialCandidate[]}
    deactivate PS

    loop For each TrialCandidate
        CLI->>CTGov: get_trial_details(nct_id) — fetch full eligibility
        CTGov-->>CLI: eligibility criteria text (inclusion + exclusion)

        CLI->>CLI: Parse eligibility into individual criteria[]

        loop For each criterion in trial eligibility
            CLI->>Val: evaluate_criterion(patient_note, criterion_text, type, adapter)
            activate Val
            Val-->>CLI: CriterionResult{MET / NOT_MET / UNKNOWN}
            deactivate Val
        end

        CLI->>Agg: aggregate_to_trial_verdict(criterion_results[])
        activate Agg
        Note over Agg: Any NOT_MET inclusion → EXCLUDED<br/>Any MET exclusion → EXCLUDED<br/>All MET inclusion + no exclusion hits → ELIGIBLE<br/>Otherwise → UNCERTAIN
        Agg-->>CLI: TrialVerdict (ELIGIBLE / EXCLUDED / NOT_RELEVANT / UNCERTAIN)
        deactivate Agg
    end

    CLI->>CLI: Rank trials by verdict + confidence
    CLI->>CLI: Save to runs/{run_id}/ with full trace
```

---

## Appendix: Key Data Structures

| Structure | Source | Purpose |
|-----------|--------|---------|
| `PresearchResult` | `prescreen/schema.py` | Agent output: candidates + trace + cost |
| `TrialCandidate` | `prescreen/schema.py` | Single trial from CT.gov with ranking metadata |
| `ToolCallRecord` | `prescreen/schema.py` | Trace record for each tool invocation |
| `CriterionResult` | `validate/evaluator.py` | Single criterion evaluation verdict + evidence |
| `TrialVerdict` | `evaluation/metrics.py` | Aggregated trial-level eligibility decision |
| `RunResult` | `models/schema.py` | Complete benchmark run for persistence |
