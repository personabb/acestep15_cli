"""Seed helpers for ``infer_retake.py``."""

import random
from numbers import Integral, Real
from typing import Protocol


MAX_RANDOM_SEED = 2**32 - 1
RetakeSeedSetting = int | list[int] | tuple[int, ...]


class _RandomSource(Protocol):
    """Protocol for random sources used during seed resolution."""

    def randint(self, start: int, end: int) -> int:
        """Return a random integer in the inclusive range ``[start, end]``."""


def _normalize_seed_value(seed_value: object) -> int:
    """Normalize one configured seed value.

    Args:
        seed_value: Raw seed value from ``RETAKE_SEED``.

    Returns:
        Normalized integer seed.

    Raises:
        ValueError: If the value is not an integer-like seed or is smaller than ``-1``.
    """
    if isinstance(seed_value, bool):
        raise ValueError("Seed value must be an integer or -1, not bool.")

    if isinstance(seed_value, Integral):
        normalized = int(seed_value)
    elif isinstance(seed_value, Real):
        numeric_value = float(seed_value)
        if not numeric_value.is_integer():
            raise ValueError(f"Seed value must be an integer or -1: {seed_value!r}")
        normalized = int(numeric_value)
    else:
        raise ValueError(f"Seed value must be an integer or -1: {seed_value!r}")

    if normalized < -1:
        raise ValueError(f"Seed value must be -1 or a non-negative integer: {seed_value!r}")
    return normalized


def serialize_seed_setting(seed_setting: RetakeSeedSetting) -> int | list[int]:
    """Return a JSON-safe representation of the configured retake seeds.

    Args:
        seed_setting: Configured ``RETAKE_SEED`` value.

    Returns:
        A scalar integer or a list of integers suitable for JSON serialization.
    """
    if isinstance(seed_setting, (list, tuple)):
        return [_normalize_seed_value(seed_value) for seed_value in seed_setting]
    return _normalize_seed_value(seed_setting)


def format_seed_setting(seed_setting: RetakeSeedSetting) -> str:
    """Format the configured retake seeds for human-readable logging.

    Args:
        seed_setting: Configured ``RETAKE_SEED`` value.

    Returns:
        String representation for console output.
    """
    return str(serialize_seed_setting(seed_setting))


def resolve_retake_seeds(
    seed_setting: RetakeSeedSetting,
    batch_size: int,
    rng: _RandomSource | None = None,
) -> list[int]:
    """Resolve configured retake seeds into a concrete per-batch seed list.

    Behavior:
      - Scalar ``>= 0`` keeps the current ``seed + i`` expansion.
      - Scalar ``-1`` resolves every batch item to a random 32-bit seed.
      - List/tuple input must match ``batch_size`` exactly; each ``-1`` entry is
        replaced with a random 32-bit seed.

    Args:
        seed_setting: Configured ``RETAKE_SEED`` value.
        batch_size: Number of tracks to generate.
        rng: Optional random source for deterministic tests.

    Returns:
        Concrete per-item seed list with no remaining ``-1`` values.

    Raises:
        ValueError: If ``batch_size`` is invalid or the seed setting is malformed.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1: {batch_size!r}")

    if isinstance(seed_setting, (list, tuple)):
        requested_seeds = serialize_seed_setting(seed_setting)
        if len(requested_seeds) != batch_size:
            raise ValueError(
                "When RETAKE_SEED is a list/tuple, its length must match "
                f"BATCH_SIZE exactly ({batch_size})."
            )
    else:
        scalar_seed = _normalize_seed_value(seed_setting)
        if scalar_seed == -1:
            requested_seeds = [-1] * batch_size
        else:
            requested_seeds = [scalar_seed + index for index in range(batch_size)]

    random_source = rng or random
    return [
        random_source.randint(0, MAX_RANDOM_SEED) if seed_value == -1 else seed_value
        for seed_value in requested_seeds
    ]
