"""CLI wrapper for the maintained SFR vegetation overlay module."""

import os
import sys


_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from O4_SFR_Vegetation_Overlay import main


if __name__ == "__main__":
    main()
