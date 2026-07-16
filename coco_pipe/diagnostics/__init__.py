"""Model-free diagnostics for learned representations."""

from .variance import (
    crossed_ss_fractions,
    nested_ss,
    null_control,
    omega_squared_from_ss,
    subject_probe,
    variance_decomposition_report,
)

__all__ = [
    "crossed_ss_fractions",
    "nested_ss",
    "null_control",
    "omega_squared_from_ss",
    "subject_probe",
    "variance_decomposition_report",
]
