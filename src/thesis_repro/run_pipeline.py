from __future__ import annotations

import argparse

from .experiments import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis fraud-detection reproduction experiments.")
    parser.add_argument("--sample-size", type=int, default=80000, help="Maximum number of rows to use.")
    args = parser.parse_args()

    results = run(sample_size=args.sample_size)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
