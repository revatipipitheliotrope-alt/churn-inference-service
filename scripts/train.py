"""Train the churn model on synthetic data and save the artifact.

Usage:
    python scripts/train.py                      # writes models/churn_model.joblib
    python scripts/train.py --rows 20000 --output /tmp/churn.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_MODEL_PATH  # noqa: E402
from app.training import generate_synthetic_customers, save_bundle, train_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000, help="synthetic rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    frame = generate_synthetic_customers(n_rows=args.rows, seed=args.seed)
    pipeline = train_pipeline(frame)
    accuracy = pipeline.score(frame[frame.columns[:-1]], frame["churned"])
    save_bundle(pipeline, args.output)
    print(f"trained on {len(frame)} rows, train accuracy {accuracy:.3f}, saved to {args.output}")


if __name__ == "__main__":
    main()
