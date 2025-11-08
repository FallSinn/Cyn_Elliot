"""Launcher script for the Synn Core assistant.

Build for distribution with:

    pyinstaller --onefile --noconsole run_synn.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from .core import SynnConfig, SynnCore


def load_config(path: Path) -> SynnConfig:
    """Load configuration from JSON file."""

    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return SynnConfig(**data)


async def main(config_path: Path) -> None:
    """Entrypoint for asynchronous execution."""

    config = load_config(config_path)
    core = SynnCore(config)
    try:
        await core.start()
    except KeyboardInterrupt:
        logging.getLogger("SynnCore").info("Keyboard interrupt received. Shutting down...")
        core.request_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Synn Core assistant")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json", help="Path to configuration file")
    args = parser.parse_args()
    asyncio.run(main(args.config))
