"""CLI wrapper for image-only overlay debug previews."""

import os
import sys


_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from O4_Overlay_Debug_Images import main


if __name__ == "__main__":
    raise SystemExit(main())
