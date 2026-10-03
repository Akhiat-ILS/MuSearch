"""Compatibility entry point; implementation lives in musearch.summarize_heldout."""

from musearch.summarize_heldout import main, paired_statistics

__all__ = ["main", "paired_statistics"]

if __name__ == "__main__":
    main()
