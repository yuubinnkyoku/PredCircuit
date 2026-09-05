#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from predcircuit.connectome import (
    MALCNS_ANNOTATIONS_URL,
    MALCNS_CONNECTIVITY_URL,
    MALCNS_NEUROTRANSMITTER_URL,
)

FILES = {
    "annotations": MALCNS_ANNOTATIONS_URL,
    "neurotransmitters": MALCNS_NEUROTRANSMITTER_URL,
    "connectivity": MALCNS_CONNECTIVITY_URL,
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"exists: {dst}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading {url}\n       -> {dst}")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed official Janelia URLs
    tmp.replace(dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Download official MaleCNS v1.0 flat-connectome files")
    p.add_argument(
        "kinds",
        nargs="*",
        choices=sorted(FILES),
        default=["annotations"],
        help="Default: annotations. Connectivity is about 1.1 GB.",
    )
    p.add_argument("--dir", type=Path, default=Path("data/raw/malecns-v1.0"))
    args = p.parse_args()
    for kind in args.kinds:
        url = FILES[kind]
        download(url, args.dir / url.rsplit("/", 1)[-1])


if __name__ == "__main__":
    main()
