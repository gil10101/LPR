#!/usr/bin/env python3
"""Generate the synthetic license-plate dataset (offline, no network).

Usage:
    python scripts/generate_synthetic.py                 # counts from config.yaml
    python scripts/generate_synthetic.py --train 2000 --val 400 --test 400
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.config import load_config
from lpr.data.synthetic import generate_dataset


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=cfg.abspath(cfg.synthetic.out_dir))
    ap.add_argument("--train", type=int, default=cfg.synthetic.num_train)
    ap.add_argument("--val", type=int, default=cfg.synthetic.num_val)
    ap.add_argument("--test", type=int, default=cfg.synthetic.num_test)
    ap.add_argument("--seed", type=int, default=cfg.synthetic.seed)
    args = ap.parse_args()

    print(f"[synthetic] generating train={args.train} val={args.val} "
          f"test={args.test} -> {args.out_dir}")
    summary = generate_dataset(args.out_dir, args.train, args.val, args.test,
                               args.seed)
    print(f"[synthetic] done: {summary}")


if __name__ == "__main__":
    main()
