"""Run the one-use physical analysis on the sealed TASK3 train/val artifacts."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/"src")]

if __name__ == "__main__":
    from kron_reduce.workflows.postfault_m4_m3b import main

    main("m4")
