"""CLI entry point: `baysbench`.

Usage
-----
Run all tasks in a benchmark module::

    baysbench my_benchmark.py

The module must expose a ``bench`` variable that is a
:class:`~baysbench.benchmark.BayesianBenchmark` instance, or a class decorated
with :func:`~baysbench.decorators.suite` whose class variable name ends with
``Benchmark`` or ``Suite`` (case-insensitive).

Options::

    baysbench my_benchmark.py --confidence 0.99 --min-samples 5

"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .benchmark import BayesianBenchmark


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_baysbench_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _find_bench(mod) -> BayesianBenchmark | None:
    """Try to locate a BayesianBenchmark instance in the loaded module."""
    # 1. Explicit `bench` variable
    if hasattr(mod, "bench") and isinstance(mod.bench, BayesianBenchmark):
        return mod.bench

    # 2. Any BayesianBenchmark instance in module globals
    for val in vars(mod).values():
        if isinstance(val, BayesianBenchmark):
            return val

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="baysbench",
        description="Bayesian sequential benchmarking for LLMs and agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "module",
        nargs="?",
        help="Path to a Python file containing benchmark definitions.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        metavar="P",
        help="Override stopping confidence threshold (default: from benchmark or 0.95).",
    )
    parser.add_argument(
        "--skip-threshold",
        type=float,
        default=None,
        metavar="P",
        help="Override non-discriminating skip threshold (default: from benchmark or 0.85).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override minimum samples before early stopping (default: from benchmark or 3).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )

    args = parser.parse_args(argv)

    if args.version:
        from importlib.metadata import version

        print(version("baysbench"))
        return 0

    if not args.module:
        parser.print_help()
        return 0

    path = Path(args.module)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        mod = _load_module(path)
    except Exception as exc:
        print(f"Error loading {path}: {exc}", file=sys.stderr)
        return 1

    # Check for @suite-decorated classes first
    suite_classes = [
        v
        for v in vars(mod).values()
        if isinstance(v, type) and hasattr(v, "_bench") and callable(getattr(v, "run", None))
    ]

    if suite_classes:
        for cls in suite_classes:
            print(f"\n--- {cls.__name__} ---")
            report = cls.run()
            print(report.summary())
        return 0

    bench = _find_bench(mod)
    if bench is None:
        print(
            "Error: no BayesianBenchmark instance found in the module.\n"
            "Define `bench = BayesianBenchmark()` or use the @suite decorator.",
            file=sys.stderr,
        )
        return 1

    # Apply CLI overrides
    if args.confidence is not None:
        bench.confidence = args.confidence
    if args.skip_threshold is not None:
        bench.skip_threshold = args.skip_threshold
    if args.min_samples is not None:
        bench.min_samples = args.min_samples

    report = bench.run()
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
