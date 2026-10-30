"""Download checksum-verified release assets and install without overwriting changes."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", default="tulay-hub/supermimicplus")
    parser.add_argument("--tag", default="v0.1.0-30000")
    parser.add_argument("--only", choices=["all", "data", "models", "assets"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release-manifest.json").read_text())
    cache = root / "release-local/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest["assets"]:
        if args.only not in ("all", item["kind"]):
            continue
        path = cache / item["name"]
        if not path.exists():
            url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{item['name']}"
            temporary = path.with_suffix(".part")
            request = urllib.request.Request(
                url, headers={"User-Agent": "supermimicplus-release-installer"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with tarfile.open(path) as archive:
            for member in archive:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts or member.issym() or member.islnk():
                    raise ValueError(f"Unsafe archive member: {member.name}")
                target = root / str(rel)
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Archive target escapes project: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                with archive.extractfile(member) as stream:
                    content = stream.read()
                if target.exists() and target.read_bytes() != content:
                    raise FileExistsError(f"Refusing to overwrite: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(f"Verified and installed {item['name']}", flush=True)


if __name__ == "__main__":
    main()
