# TrialMatch (MedGemma Demo)

Clinical trial matching demo and benchmark dashboard for MedGemma + Gemini workflows.

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Set required environment variables:

```bash
export GOOGLE_API_KEY="..."
export HF_TOKEN="..."
export GCP_PROJECT_ID="..."
export GCP_REGION="us-central1"
export VERTEX_ENDPOINT_ID="..."       # MedGemma 4B imaging endpoint
export VERTEX_ENDPOINT_ID_27B="..."   # MedGemma 27B reasoning endpoint
```

If Vertex endpoint vars are present, runtime routes MedGemma calls as:
- imaging tasks -> Vertex MedGemma 4B (`VERTEX_ENDPOINT_ID`)
- medical reasoning tasks -> Vertex MedGemma 27B (`VERTEX_ENDPOINT_ID_27B`)

Set `TRIALMATCH_FORCE_HF_MEDGEMMA=1` to force HF MedGemma fallback.

3. Generate demo-safe cached artifacts for curated patients:

```bash
uv run python scripts/demo/generate_cached_runs.py
```

4. Launch Streamlit:

```bash
uv run streamlit run demo/app.py --server.port 8501
```

## Demo Recording Modes

- `cached` mode (recommended for recording): deterministic replay from `demo/data/cached_runs/`.
- `live` mode: runs preflight checks against Gemini, MedGemma, and CT.gov. Any failed preflight blocks execution.

Curated cached patients:

- `mpx1016`
- `mpx1575`
- `mpx1875`

## Benchmark Dashboard

Benchmark page loads **pinned comparable runs** from:

- `demo/data/benchmark/pinned_runs.json`

Pinned runs must share:

- same `data.seed`
- same `(patient_id, trial_id, pair_index)` set hash

## QA Commands

```bash
uv run pytest tests/unit
uv run pytest -m "not smoke and not e2e"
```

See `docs/qa/demo-runbook.md` for full reproduction and smoke QA flow.
