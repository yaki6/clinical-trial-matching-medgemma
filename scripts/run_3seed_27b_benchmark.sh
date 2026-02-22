#!/usr/bin/env bash
# Run 3 fresh-seed two-stage benchmarks (27B + Flash) sequentially
set -euo pipefail

echo "=== Running 3 fresh-seed benchmarks (seeds 7, 99, 256) ==="
echo ""

echo "--- Seed 7 ---"
uv run trialmatch phase0 --config configs/phase0_two_stage_27b_flash_seed7.yaml
echo ""

echo "--- Seed 99 ---"
uv run trialmatch phase0 --config configs/phase0_two_stage_27b_flash_seed99.yaml
echo ""

echo "--- Seed 256 ---"
uv run trialmatch phase0 --config configs/phase0_two_stage_27b_flash_seed256.yaml
echo ""

echo "=== All 3 seeds complete ==="
