"""CLI entry point for the final 3.10 paper comparison.

The implementation is organized under:
  - our_methods/ for M0, M2A, and M2A-LYAP checks/retraining helpers
  - comparison_methods/ for external baseline entry points
  - support/ for shared orchestration, reporting, and audits
"""

from __future__ import annotations

from support.pipeline import main


if __name__ == "__main__":
    main()
