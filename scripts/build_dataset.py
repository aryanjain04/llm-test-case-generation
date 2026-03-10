"""
Mine GitHub repos and build training dataset.

Usage:
    python scripts/build_dataset.py --max-repos 5   # start small
    python scripts/build_dataset.py --max-repos 20  # full dataset
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import GitHubMiner, DatasetBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-repos", type=int, default=5)
    parser.add_argument("--clone-dir", default="repos_cache")
    parser.add_argument("--output", default="datasets/train.jsonl")
    parser.add_argument("--skip-clone", action="store_true", help="Use already cloned repos")
    args = parser.parse_args()

    # 1. Clone repos
    miner = GitHubMiner(clone_dir=args.clone_dir)

    if not args.skip_clone:
        print(f"=== Cloning up to {args.max_repos} repos ===")
        miner.clone_repos(max_repos=args.max_repos)

    repos = miner.get_repos()
    print(f"\nFound {len(repos)} repos to process\n")

    # 2. Extract function-test pairs
    builder = DatasetBuilder(output_dir=str(Path(args.output).parent))

    for repo_path in repos:
        print(f"Processing {repo_path.name}...")
        pairs = builder.process_repo(str(repo_path))
        print(f"  Found {len(pairs)} function-test pairs")

    # 3. Save dataset
    output_name = Path(args.output).name
    builder.save(output_name)

    # 4. Stats
    print("\n=== Dataset Statistics ===")
    builder.stats()


if __name__ == "__main__":
    main()
