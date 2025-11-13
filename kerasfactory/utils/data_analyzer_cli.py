#!/usr/bin/env python
"""Command-line interface for the Keras Model Registry Data Analyzer.

This script provides a convenient way to analyze CSV data and get layer recommendations
from the command line.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any
from loguru import logger
from kerasfactory.utils.data_analyzer import DataAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze CSV data and recommend kerasfactory layers for model building",
    )

    parser.add_argument(
        "source",
        type=str,
        help="Path to CSV file or directory containing CSV files",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match when source is a directory (default: *.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the JSON output (default: print to stdout)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--recommendations-only",
        action="store_true",
        help="Only output layer recommendations without detailed statistics",
    )

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity.

    Args:
        verbose: Whether to enable verbose logging
    """
    logger.remove()  # Remove default handlers

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def format_result(result: dict[str, Any], recommendations_only: bool) -> dict[str, Any]:
    """Format the result based on user preferences.

    Args:
        result: The analysis result
        recommendations_only: Whether to include only recommendations

    Returns:
        Formatted result dictionary
    """
    if recommendations_only:
        return {"recommendations": result.get("recommendations", {})}
    return result


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check if source exists
    if not Path(args.source).exists():
        logger.error(f"Source not found: {args.source}")
        sys.exit(1)

    try:
        # Create analyzer and run analysis
        analyzer = DataAnalyzer()
        result = analyzer.analyze_and_recommend(args.source, args.pattern)

        # Format result
        formatted_result = format_result(result, args.recommendations_only)

        # Output result
        if args.output:
            with Path(args.output).open("w") as f:
                json.dump(formatted_result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print to stdout
            print(json.dumps(formatted_result, indent=2))

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        if args.verbose:
            logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
