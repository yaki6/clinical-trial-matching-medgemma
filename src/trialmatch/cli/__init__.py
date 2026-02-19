"""CLI entry point for trialmatch."""

import click

from trialmatch.cli.phase0 import phase0_cmd


@click.group()
def main():
    """MedGemma vs Gemini 3 Pro: Clinical trial criterion matching benchmark."""
    pass


main.add_command(phase0_cmd)
