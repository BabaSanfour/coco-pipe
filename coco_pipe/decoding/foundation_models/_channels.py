"""Channel-name normalization for EEG foundation models.

A dependency-free leaf module: it imports nothing from the package, so any
module (including ``_montages``, whose constants feed the capability gate) can
import it at module level without an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

# Legacy 10-20 temporal labels -> modern names.
_CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_channel_names(ch_names: Sequence[str]) -> list[str]:
    """Normalize legacy 10-20 temporal labels (T3->T7, T4->T8, T5->P7, T6->P8)."""
    return [_CHANNEL_ALIASES.get(str(name), str(name)) for name in ch_names]
