"""Run the registered build or the one-use evaluation for M5."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "src")]

if __name__ == "__main__":
    from kron_reduce.workflows.blind_confirmation_m5 import main

    main()
