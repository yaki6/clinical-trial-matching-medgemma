# Vertex Deploy Status (2026-02-22)

## Goal
Use Vertex MedGemma `4B` for imaging tasks and Vertex MedGemma `27B` for medical reasoning in live demo cache generation.

## Current Status

### 27B reasoning
- Endpoint: `7499477442478735360`
- Deploy operation: `7613146887293501440`
- Result: `SUCCESSFULLY_DEPLOYED`
- Health: `True` via `trialmatch.live_runtime.create_reasoning_adapter(...).health_check()`

### 4B imaging
- Endpoint in `.env`: `1501808638728077312`
- Current result: **NOT READY** (`traffic_split not set`, no deployed models)

## 4B Deploy Attempts and Outcomes

1. `5818040223321292800` (T4, n1-standard-8)
- Failed: `FAILED_TO_DEPLOY`
- Log root cause: swap-space mismatch (`32 GiB swap` on `29.38 GiB` RAM)

2. `2052467984886136832` (V100 candidate on endpoint `1501808638728077312`)
- Failed: `FAILED_TO_DEPLOY`

3. `2791621273728319488` (T4, n1-standard-16 candidate on endpoint `1501808638728077312`)
- Still running in `STARTING_MODEL_SERVER` for extended duration

4. New endpoint `923518299076034560` created for clean 4B deploy
- `6826846539852283904` failed (`CustomModelServingCPUsPerProjectPerRegion`, `CustomModelServingT4GPUsPerProjectPerRegion`)
- `6686214604613222400` (custom 4B model on V100) still in `ADDING_NODES_TO_CLUSTER`
- `2989216707379200000` / `3493619865644695552` failed (`CustomModelServingCPUsPerProjectPerRegion`)

## Root-Cause Summary

1. Existing non-cancellable deploy operations consume custom-serving CPU/T4 quota, blocking fresh 4B attempts.
2. Default 4B model artifact runs with args unsuitable for some 1-GPU profiles (swap and tensor-parallel behavior), requiring custom serving args.

## Impact

- Live E2E cache generation now passes preflight for:
  - Gemini
  - CT.gov
  - 27B reasoning endpoint
- It still fails at first image call because 4B imaging endpoint is not ready.

## Gate to Clear

- At least one 4B Vertex endpoint must pass health check:
  - `trialmatch.live_runtime.create_imaging_adapter(...).health_check() == True`
- Then rerun:
  - `uv run python scripts/demo/generate_cached_runs.py --topics mpx1016 mpx1575 mpx1875`

