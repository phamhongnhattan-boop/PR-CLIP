"""Fetch and safely unpack the image archive linked by the official PR-CLIP repo."""

import argparse
import hashlib
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen


FILE_ID = "15htMe2iE4kVujjWURxfCxZXk8vAAetoT"
EXPECTED_BYTES = 1_461_515_987
EXPECTED_SHA256 = "f61d5d59e5e0696cb88685c34c2ee58961d68174e76a6988c9a83b2bb38f11d5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("images"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    archive = args.directory / "images.tar.gz"
    if not archive.exists() or archive.stat().st_size != EXPECTED_BYTES:
        url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=120) as response, archive.open("wb") as target:
            if response.headers.get("content-type") != "application/octet-stream":
                raise RuntimeError("Google Drive did not return the image archive")
            copied = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                copied += len(chunk)
                if copied % (100 * 1024 * 1024) < len(chunk):
                    print(f"downloaded {copied / 1e9:.2f} GB", flush=True)
        if archive.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError("Image archive download was incomplete")
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Image archive SHA-256 mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "image_all":
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
        print(f"validated {len(members)} entries; extracting", flush=True)
        # The member check above is used because Windows sandboxing can reject
        # tarfile's data_filter realpath check for otherwise valid paths.
        tar.extractall(args.directory, members=members, filter="fully_trusted")
    print("image directory:", (args.directory / "image_all").resolve())


if __name__ == "__main__":
    main()
