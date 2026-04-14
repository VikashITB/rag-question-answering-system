#!/usr/bin/env python3
"""
run.py — Development server launcher.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8000 --reload

For production, use:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
    (Note: Multiple workers won't share the in-memory registry.
     Use a DB-backed registry for multi-worker production deployments.)
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG QA System server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable hot-reload for development (do NOT use in production)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting RAG QA System on http://{args.host}:{args.port}")
    print(f"📖 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔄 Reload:   {'enabled' if args.reload else 'disabled'}\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
