"""Base protocol for model adapters.

This interface is shared by both benchmark and e2e pipeline model usage.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trialmatch.models.schema import ModelResponse


class ModelAdapter(abc.ABC):
    """Abstract base for LLM model adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Model name for logging and run tracking."""

    @abc.abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Send prompt to model and return structured response."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if model endpoint is reachable."""
