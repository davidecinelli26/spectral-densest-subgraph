"""The 15 SNAP graphs of the thesis: download, checksum, parsing.

Every graph is read as a simple undirected graph (direction, self-loops,
parallel edges and edge attributes are dropped), from the files published by
SNAP (https://snap.stanford.edu/data/).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from densest.graph import Graph

SNAP = "https://snap.stanford.edu/data/"


@dataclass(frozen=True)
class Source:
    archive: str  # file name on SNAP
    sha256: str
    member: str | None = None  # file inside a .zip archive
    description: str = ""


DATASETS: dict[str, Source] = {
    "as20000102": Source(
        "as20000102.txt.gz", "b5bbd9318d2e3d7fa64be29068f53d5553f0af4729cda4e53203a125f1e414f0",
        description="Autonomous systems, BGP logs of 2 January 2000",
    ),
    "as-caida20071105": Source(
        "as-caida20071105.txt.gz", "34ce3d63dc03bbbabbb05fdb15d5cb7fe52564166d20f860755ca0e794ac057a",
        description="CAIDA AS relationships, 5 November 2007",
    ),
    "ca-AstroPh": Source(
        "ca-AstroPh.txt.gz", "51bf1e2cace269b884481a8502474efa67c0fd01d998ff7f5a154d7d3e527f27",
        description="arXiv Astro Physics co-authorship",
    ),
    "ca-GrQc": Source(
        "ca-GrQc.txt.gz", "a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230",
        description="arXiv General Relativity co-authorship",
    ),
    "deezer_europe": Source(
        "deezer_europe.zip", "dd66a73f8d8690b5bc300ba378883fb2c2f6316aec8917b6a2428e352fc9e498",
        member="deezer_europe/deezer_europe_edges.csv", description="Deezer users from Europe",
    ),
    "email-Eu-core": Source(
        "email-Eu-core.txt.gz", "4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f",
        description="E-mails within a European research institution",
    ),
    "facebook_combined": Source(
        "facebook_combined.txt.gz", "125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
        description="Facebook social circles",
    ),
    "lastfm_asia": Source(
        "lastfm_asia.zip", "51acb78a923bb223ed6e61be88f91122fb29adca3f07beff7289cafd98601d47",
        member="lasftm_asia/lastfm_asia_edges.csv", description="LastFM users from Asia",
    ),
    "musae_chameleon": Source(
        "wikipedia.zip", "aeef409fca6b08abe9e3751bf41839b2544d499f0eca174701f87c9f93c71b86",
        member="wikipedia/chameleon/musae_chameleon_edges.csv", description="Wikipedia pages on chameleons",
    ),
    "musae_squirrel": Source(
        "wikipedia.zip", "aeef409fca6b08abe9e3751bf41839b2544d499f0eca174701f87c9f93c71b86",
        member="wikipedia/squirrel/musae_squirrel_edges.csv", description="Wikipedia pages on squirrels",
    ),
    "oregon1_010526": Source(
        "oregon1_010526.txt.gz", "bc977801fa52a822f2e72d79cbb550c0ddade08540b46b3d29b898b30ee55fe0",
        description="AS peering from Oregon route-views, 26 May 2001",
    ),
    "p2p-Gnutella08": Source(
        "p2p-Gnutella08.txt.gz", "cb3019062be21e55b03f19a73e88edf34693bafb68ae581621cc7a037fa604a9",
        description="Gnutella peer-to-peer network, 8 August 2002",
    ),
    "twitch_gamers": Source(
        "twitch_gamers.zip", "9c1d4eb7c606e77c0dc8556f16db365f37e5a430b185e5414b7155334cd21558",
        member="large_twitch_edges.csv", description="Twitch users (mutual follows)",
    ),
    "twitter_combined": Source(
        "twitter_combined.txt.gz", "d9f99b0e6a53b9204b8c215f41b3c10fb99a1e1e783858c012b06d0d3d4bd129",
        description="Twitter social circles",
    ),
    "wiki-Vote": Source(
        "wiki-Vote.txt.gz", "7d3e53626e14b8b09fb3b396bece9d481ad606bd64ceab066349ff57d4ada7fc",
        description="Wikipedia adminship votes up to January 2008",
    ),
}  # fmt: skip


def download(name: str, raw_dir: str | Path = "data/raw") -> Path:
    """Download the archive of a dataset (once) and verify its checksum."""
    src = DATASETS[name]
    path = Path(raw_dir) / src.archive
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading {SNAP + src.archive}")
        urllib.request.urlretrieve(SNAP + src.archive, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != src.sha256:
        raise ValueError(f"{path} does not match the expected SHA-256 checksum")
    return path


def parse_edges(text: str, csv: bool = False) -> Graph:
    """Edge list in a SNAP format: '#' comments and whitespace (.txt), or a header line and commas (.csv).
    Columns after the first two (edge attributes) are ignored."""
    df = pd.read_csv(
        io.StringIO(text), sep="," if csv else r"\s+", comment=None if csv else "#", header=0 if csv else None,
        usecols=[0, 1], dtype="int64",
    )  # fmt: skip
    return Graph.from_edges(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())


def load(name: str, raw_dir: str | Path = "data/raw") -> Graph:
    src = DATASETS[name]
    path = download(name, raw_dir)
    if src.member:
        with zipfile.ZipFile(path) as archive:
            text = archive.read(src.member).decode()
    else:
        text = gzip.decompress(path.read_bytes()).decode()
    return parse_edges(text, csv=src.member is not None)
