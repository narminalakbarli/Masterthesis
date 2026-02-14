from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .experiments import run


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis fraud-detection reproduction experiments.")
    parser.add_argument("--sample-size", type=int, default=None, help="Override maximum number of rows to use.")
    parser.add_argument("--seed", type=int, default=None, help="Override global random seed for reproducible runs.")
    parser.add_argument("--config", type=str, default="config/experiment_config.json", help="Path to JSON config file.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    results = run(sample_size=args.sample_size, random_state=args.seed, config=cfg)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
