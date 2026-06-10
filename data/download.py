"""Download specific subfolders from the polysight dataset on Hugging Face.

Usage:
    python download.py --list                  # List available folders
    python download.py materials               # Download the materials folder
    python download.py labels raw              # Download multiple folders
    python download.py labels/20250603*        # Download matching sessions
    python download.py --all                   # Download everything
"""

import argparse
import fnmatch
import sys

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "xinghs/polysight"
REPO_TYPE = "dataset"


def list_folders(api: HfApi) -> list[str]:
    top_level = set()
    second_level: dict[str, set[str]] = {}
    for item in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True):
        if type(item).__name__ != "RepoFolder":
            continue
        parts = item.path.split("/")
        top_level.add(parts[0])
        if len(parts) == 2:
            second_level.setdefault(parts[0], set()).add(parts[1])

    result = []
    for t in sorted(top_level):
        result.append(t)
        for s in sorted(second_level.get(t, [])):
            result.append(f"{t}/{s}")
    return result


def resolve_patterns(patterns: list[str], available: list[str]) -> list[str]:
    matched = []
    for pat in patterns:
        pat_clean = pat.rstrip("/")
        hits = [f for f in available if fnmatch.fnmatch(f, pat_clean) or f.startswith(pat_clean + "/") or f == pat_clean]
        if not hits:
            print(f"Warning: no match for '{pat}'")
        matched.extend(hits)
    return sorted(set(matched))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", help="Folders or glob patterns to download")
    parser.add_argument("--list", action="store_true", help="List available folders")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    api = HfApi()

    if args.list:
        print(f"Available folders in {REPO_ID}:\n")
        folders = list_folders(api)
        current_top = None
        for f in folders:
            if "/" not in f:
                current_top = f
                print(f"  {f}/")
            else:
                print(f"    {f.split('/', 1)[1]}/")
        return

    if not args.folders and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        print(f"Downloading entire dataset from {REPO_ID}...")
        snapshot_download(
            REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=".",
        )
        print("Done.")
        return

    available = list_folders(api)
    targets = resolve_patterns(args.folders, available)

    if not targets:
        print("No matching folders found.")
        sys.exit(1)

    print(f"Downloading {len(targets)} folder(s):")
    for t in targets:
        print(f"  {t}/")

    allow = [f"{t}/**" for t in targets] + [f"{t}/*" for t in targets]

    snapshot_download(
        REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=".",
        allow_patterns=allow,
    )
    print("Done.")


if __name__ == "__main__":
    main()
