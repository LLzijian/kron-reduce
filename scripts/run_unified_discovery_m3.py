"""Run the registered synthetic gate and one-use train/val discovery."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/"src")]

if __name__ == "__main__":
    from kron_reduce.workflows.unified_discovery_m3 import main

    main()
