"""Fetches PopSign ASL v1.0 archives and unpacks them into the training layout.

    python ml/popsign.py fetch dog fish car --out data/popsign/tars --jobs 2
    python ml/popsign.py unpack dog fish car --tars data/popsign/tars \
        --out data/popsign/raw --per-signer 3

PopSign ASL v1.0 is CC BY 4.0 and public (signdata.cc.gatech.edu, no login). Each sign
is one tar per split, 0.03 to 3.2 GB, so downloads resume instead of restarting.

Sign names are the server's folder names and are case-sensitive: almost all are
lowercase (`dog`, `thankyou`) but a few are not (`TV`; `tv` is a 404).

PopSign's own train/val/test split is ignored on purpose. Clips are pooled across the
three archives and capped per signer so every signer contributes evenly, then
`ml/make_split.py` re-splits by signer.
"""
from __future__ import annotations

import argparse
import http.client
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

BASE_URL = "https://signdata.cc.gatech.edu/data/popsign_v1_0/game"
SPLITS = ("train", "val", "test")
CHUNK = 1 << 20

# `4a.7005-dog-2023_01_03_15_08_59.922-0.mp4` and `gtsignstudy4a.8045-dog-...` both
# carry the signer after `4a.`; it is the only place the signer is recorded.
_SIGNER = re.compile(r"4a\.(\d+)-")

# Archive members and sign names end up in file paths, so both are allow-listed
# rather than cleaned. PurePosixPath does not treat `\` as a separator, and on Windows
# a member called `x/..\..\evil.mp4` would otherwise be written outside the raw root.
_CLIP = re.compile(r"[A-Za-z0-9_.\-]+\.mp4", re.IGNORECASE)
_SIGN = re.compile(r"[A-Za-z0-9_\-]+")


def valid_sign(sign: str) -> bool:
    """True for a plain folder name: no separators, dots, colons or wildcards."""
    return bool(_SIGN.fullmatch(sign))


def signer_id(name: str) -> str | None:
    """The signer id in a PopSign clip file name, or None when it has none."""
    match = _SIGNER.search(name)
    return match.group(1) if match else None


def select(names, existing, per_signer: int) -> list[str]:
    """Which of `names` to add so no signer exceeds `per_signer` clips of this sign.

    `existing` are the clip file names already unpacked; they count toward the cap,
    which is what makes a second run add nothing. Names are taken in sorted order so
    the choice does not depend on the order clips sit in the archives. A clip without
    a signer id is never chosen: it could not be split by signer.
    """
    existing = set(existing)
    have = Counter(signer_id(name) for name in existing)
    chosen = []
    for name in sorted(set(names) - existing):
        signer = signer_id(name)
        if signer is None or have[signer] >= per_signer:
            continue
        have[signer] += 1
        chosen.append(name)
    return chosen


# ── fetch ─────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, opener=urllib.request.urlopen, retries: int = 5,
             pause: float = 5.0) -> tuple[str, int]:
    """Downloads `url` to `dest`, resuming a partial file. Returns (status, bytes).

    Status is "ok", "exists", "missing" (HTTP 404) or "failed: <why>". Data goes to
    `<dest>.part` and is renamed only when complete and readable to its end as a tar,
    so a file named `.tar` is always a whole archive: an error page or a dropped
    connection can never be mistaken for one by the unpack step.

    Dropped connections, HTTP 429 and 5xx are retried up to `retries` times, `pause`
    seconds apart; any other HTTP error fails at once, since asking again cannot help.
    """
    dest = Path(dest)
    if dest.exists():
        return "exists", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    why = "no attempt made"
    for attempt in range(retries):
        if attempt:
            time.sleep(pause)
        have = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with opener(request, timeout=60) as response:
                # 200 to a Range request means the server ignored it and is sending
                # the whole file, so the partial file must be overwritten, not extended.
                resumed = response.status == 206
                length = response.headers.get("Content-Length")
                expected = (have if resumed else 0) + int(length) if length else None
                with open(part, "ab" if resumed else "wb") as out:
                    shutil.copyfileobj(response, out, CHUNK)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "missing", 0
            if error.code == 416 and have:
                # Nothing past our offset: either the partial file is already whole
                # (a run died between the last byte and the rename) or it is stale.
                total = (error.headers.get("Content-Range") or "").rpartition("/")[2]
                if total == str(have):
                    break
                part.unlink()
                why = "partial file did not match the server's; discarded"
                continue
            if error.code == 429 or error.code >= 500:           # busy or briefly broken
                why = f"HTTP {error.code}"
                continue
            return f"failed: HTTP {error.code}", 0
        except (OSError, http.client.HTTPException) as error:   # timeout, reset, short read
            why = f"{type(error).__name__}: {error}"
            continue
        if expected is not None and part.stat().st_size < expected:
            why = "connection dropped"
            continue
        break
    else:
        return f"failed: {why} (run again to resume)", 0

    # Walk every member header, not just the first: without a Content-Length a cut-off
    # download is only detectable by the archive ending in the middle of a clip.
    try:
        with tarfile.open(part) as tar:
            for _ in tar:
                pass
    except (tarfile.TarError, EOFError) as error:
        part.unlink()
        return f"failed: not a complete tar archive ({error}); discarded", 0
    part.replace(dest)
    return "ok", dest.stat().st_size


def fetch(signs, out: Path, jobs: int) -> int:
    """Downloads every sign's three archives; returns the number that failed."""
    targets = [(split, sign) for sign in signs for split in SPLITS]
    failed, missing = 0, []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(download, f"{BASE_URL}/{split}/{sign}.tar", Path(out) / split / f"{sign}.tar"):
                (split, sign)
            for split, sign in targets
        }
        for n, future in enumerate(as_completed(futures), 1):
            split, sign = futures[future]
            status, size = future.result()
            failed += status.startswith("failed")
            if status == "missing":
                missing.append(f"{split}/{sign}")
                status = "404 not found, skipped"
            print(f"[{n}/{len(targets)}] {split}/{sign}.tar  {size / 1e6:.1f} MB  {status}", flush=True)
    if missing:
        print(f"\nnot on the server: {', '.join(sorted(missing))}\n"
              "sign names are case-sensitive folder names (e.g. `TV`, not `tv`)")
    return failed


# ── unpack ────────────────────────────────────────────────────────────────────

def unpack_sign(sign: str, tars: Path, raw: Path, per_signer: int) -> dict:
    """Extracts up to `per_signer` clips per signer of one sign into raw/<signer>/<sign>/.

    Returns counts for the summary line. Only the file name of an archive member is
    used, never its path, and only when it is a plain clip name, so a member cannot be
    written outside `raw`. An archive that cannot be read to its end contributes
    nothing and is named in `unreadable`; the other splits are still unpacked.
    """
    if not valid_sign(sign):
        raise ValueError(f"not a plain sign name: {sign!r}")
    tars, raw = Path(tars), Path(raw)
    archives = [tars / split / f"{sign}.tar" for split in SPLITS]
    absent = [a.parent.name for a in archives if not a.exists()]

    where = {}                      # clip file name -> (archive, member name)
    unreadable = []
    for archive in archives:
        if not archive.exists():
            continue
        found = {}
        try:
            with tarfile.open(archive) as tar:
                for member in tar:
                    name = PurePosixPath(member.name).name
                    if member.isfile() and _CLIP.fullmatch(name):
                        found[name] = (archive, member.name)
        except (tarfile.TarError, EOFError, OSError):
            unreadable.append(archive.parent.name)
            continue
        for name, source in found.items():
            where.setdefault(name, source)

    existing = [p.name for p in raw.glob(f"*/{sign}/*.mp4")]
    chosen = select(where, existing, per_signer)

    by_archive: dict[Path, list[str]] = {}
    for name in chosen:
        by_archive.setdefault(where[name][0], []).append(name)
    for archive, names in by_archive.items():
        with tarfile.open(archive) as tar:
            for name in names:
                target = raw / signer_id(name) / sign / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # Written under another suffix first: a run killed mid-clip must not
                # leave a truncated .mp4 that counts toward the cap forever.
                partial = target.with_name(name + ".part")
                with tar.extractfile(tar.getmember(where[name][1])) as source, open(partial, "wb") as out:
                    shutil.copyfileobj(source, out, CHUNK)
                partial.replace(target)

    clips = list(raw.glob(f"*/{sign}/*.mp4"))
    return {
        "added": len(chosen),
        "total": len(clips),
        "signers": len({p.parent.parent.name for p in clips}),
        "no_signer_id": sum(signer_id(name) is None for name in where),
        "absent": absent,
        "unreadable": unreadable,
    }


def unpack(signs, tars: Path, raw: Path, per_signer: int) -> int:
    """Unpacks each sign and prints its summary line; returns how many signs had no
    archive or an unreadable one."""
    failed = 0
    for sign in signs:
        r = unpack_sign(sign, tars, raw, per_signer)
        if len(r["absent"]) == len(SPLITS):
            print(f"{sign}: no archives under {tars} (run fetch first)", flush=True)
            failed += 1
            continue
        notes = ""
        if r["unreadable"]:
            failed += 1
            notes += f"  (UNREADABLE archive for: {', '.join(r['unreadable'])}; delete it and fetch again)"
        if r["absent"]:
            notes += f"  (no archive for: {', '.join(r['absent'])})"
        if r["no_signer_id"]:
            notes += f"  ({r['no_signer_id']} clips without a signer id left out)"
        print(f"{sign}: +{r['added']} clips, {r['total']} total, {r['signers']} signers{notes}", flush=True)
    return failed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download each sign's train/val/test archives (resumable)")
    f.add_argument("signs", nargs="+")
    f.add_argument("--out", required=True, help="archives land in <out>/<split>/<sign>.tar")
    f.add_argument("--jobs", type=int, default=2, help="parallel downloads")

    u = sub.add_parser("unpack", help="extract a capped number of clips per signer per sign")
    u.add_argument("signs", nargs="+")
    u.add_argument("--tars", required=True, help="the directory given to fetch --out")
    u.add_argument("--out", required=True, help="raw root: <out>/<signer>/<sign>/<clip>.mp4")
    u.add_argument("--per-signer", type=int, default=3)

    args = ap.parse_args(argv)
    odd = [sign for sign in args.signs if not valid_sign(sign)]
    if odd:
        raise SystemExit(f"not plain sign names (letters, digits, `_`, `-` only): {odd}")
    if args.command == "fetch":
        failed = fetch(args.signs, Path(args.out), args.jobs)
    else:
        failed = unpack(args.signs, Path(args.tars), Path(args.out), args.per_signer)
    if failed:
        raise SystemExit(f"{failed} failed")


if __name__ == "__main__":
    main()
