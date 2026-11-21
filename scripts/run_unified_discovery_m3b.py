"""Run three paired supports or the separately gated fixed test."""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/"src")]

if __name__ == "__main__":
    from kron_reduce.workflows.postfault_m4_m3b import main

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "test"))
    main(parser.parse_args().stage)
