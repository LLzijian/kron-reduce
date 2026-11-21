"""Run the one-use, archived-only M4 v2 analysis."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/"src")]

if __name__ == "__main__":
    from kron_reduce.workflows.postfault_m4_v2 import main

    main()
