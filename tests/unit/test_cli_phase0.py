"""Tests for CLI phase0 command."""

from click.testing import CliRunner

from trialmatch.cli import main


def test_phase0_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--help"])
    assert result.exit_code == 0
    assert "phase0" in result.output.lower() or "Phase 0" in result.output


def test_phase0_dry_run_with_fixture():
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--dry-run", "--config", "configs/phase0_test.yaml"])
    assert result.exit_code == 0
    assert "Dry run" in result.output or "dry run" in result.output.lower()
