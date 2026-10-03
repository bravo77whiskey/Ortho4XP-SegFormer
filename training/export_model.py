"""CLI helper to export the best placement checkpoint."""

from __future__ import annotations

import argparse
import json

from building_training_lib import export_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export best building placement checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = export_model(args.run_dir, args.output)
    print(json.dumps({"exported": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
