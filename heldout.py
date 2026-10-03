"""Compatibility entry point; implementation lives in musearch.heldout."""

from musearch.heldout import (
    MODELS,
    PROJECT,
    CountedModel,
    digest,
    main,
    sample_indices,
    validate,
    write_json,
)

__all__ = [
    "MODELS",
    "PROJECT",
    "CountedModel",
    "digest",
    "main",
    "sample_indices",
    "validate",
    "write_json",
]

if __name__ == "__main__":
    main()
