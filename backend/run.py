"""Launcher: uvicorn backend.app:app with sane defaults.

Usage:
    python backend/run.py            # http://127.0.0.1:8000
    python backend/run.py --port 8000 --reload
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="VigiDriver backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.register("type", None, str)  # ensure str parsing
    parser.add_argument("--port", default="8000")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "app:app",
        host=args.host,
        port=int(args.port),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
