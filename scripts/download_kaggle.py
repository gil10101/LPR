#!/usr/bin/env python3
"""Download the project's Kaggle datasets using credentials from .env.

Reads KAGGLE_USERNAME and KAGGLE_KEY from a project-root ``.env`` (see
``.env.example``), then downloads and unzips the recognition + detection sets
into ``data/``:

    pip install kaggle
    cp .env.example .env      # then fill in your Kaggle username + key
    python scripts/download_kaggle.py

Credentials are loaded into the environment *before* the kaggle client is
imported, which avoids the "KeyError: 'username'" you get when the client tries
to authenticate with a half-configured token.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.utils.env import load_dotenv

# Default datasets and where they land under data/.
DATASETS = [
    ("nickyazdani/license-plate-text-recognition-dataset", "data/nicklpsr"),
    ("fareselmenshawii/large-license-plate-dataset", "data/llpd"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", default=None,
                    help="owner/name[:dest] to download (repeatable). "
                         "Defaults to the project's two datasets.")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        sys.exit("Missing Kaggle credentials. Copy .env.example to .env and set "
                 "KAGGLE_USERNAME and KAGGLE_KEY (kaggle.com -> Settings -> API).")

    # Import only after credentials are in the environment.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is not installed. Run: pip install kaggle")

    api = KaggleApi()
    api.authenticate()

    targets = []
    if args.dataset:
        for spec in args.dataset:
            if ":" in spec:
                ds, dest = spec.split(":", 1)
            else:
                ds, dest = spec, os.path.join("data", spec.split("/")[-1])
            targets.append((ds, dest))
    else:
        targets = DATASETS

    for ds, dest in targets:
        os.makedirs(dest, exist_ok=True)
        print(f"[kaggle] downloading {ds} -> {dest}")
        api.dataset_download_files(ds, path=dest, unzip=True, quiet=False)
        print(f"[kaggle] done: {dest}")

    print("[kaggle] all downloads complete.")


if __name__ == "__main__":
    main()
