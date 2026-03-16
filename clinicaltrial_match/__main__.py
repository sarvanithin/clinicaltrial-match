"""Entry point: python -m clinicaltrial_match serve"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctm", description="clinicaltrial-match CLI")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "serve":
        _serve(args)
    else:
        parser.print_help()
        sys.exit(1)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn
    from clinicaltrial_match.config import get_config
    from clinicaltrial_match.api.app import create_app

    config = get_config()
    host = args.host or config.api.host
    port = args.port or config.api.port

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=args.reload,
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()
