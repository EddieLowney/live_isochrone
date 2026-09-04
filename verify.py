#!/usr/bin/env python3
"""Check every tracked file against MANIFEST.sha256.

Applying updates file by file has repeatedly gone wrong — a browser renames a
repeat download, one `mv` silently moves a stale copy, and the result is a tree
that is mostly current and subtly not. This says exactly which files are wrong.
"""
import hashlib
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent
manifest = root / "MANIFEST.sha256"
if not manifest.is_file():
    sys.exit("MANIFEST.sha256 not found — run this from the project root")

stale, missing, ok = [], [], 0
for line in manifest.read_text().splitlines():
    if not line.strip():
        continue
    want, name = line.split(None, 1)
    path = root / name.strip()
    if not path.is_file():
        missing.append(name.strip())
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if got == want:
        ok += 1
    else:
        stale.append(name.strip())

print(f"{ok} current, {len(stale)} stale, {len(missing)} missing")
for n in stale:
    print(f"  STALE    {n}")
for n in missing:
    print(f"  MISSING  {n}")
if stale or missing:
    print("\nRe-extract the zip: unzip -o <archive> -d ~ replaces everything at once.")
    sys.exit(1)
print("tree matches the manifest")
