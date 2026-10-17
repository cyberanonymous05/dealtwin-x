"""Write MANIFEST.sha256: a SHA-256 hash of every raw-cache file, processed data file, embedding and
result artifact, so the exact snapshot behind the paper's numbers can be verified with
`sha256sum -c MANIFEST.sha256`."""

import hashlib
import os

ROOTS = ["data/raw", "data/processed", "data/embeddings", "results"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for root in ROOTS:
        for dirpath, _, files in os.walk(root):
            for name in sorted(files):
                if name.endswith(".tmp"):
                    continue
                path = os.path.join(dirpath, name)
                lines.append(f"{sha256(path)}  {path}")
    lines.sort(key=lambda l: l.split("  ", 1)[1])
    with open("MANIFEST.sha256", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"hashed {len(lines)} files")


if __name__ == "__main__":
    main()
