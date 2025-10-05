#!/usr/bin/env python3
"""
Backup CSV files before running trading scripts.

Usage:
    python backup_csvs.py [--data-dir "directory"]

Creates timestamped backups in backups/ directory.
"""
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path


def backup_csvs(data_dir: str = ".") -> None:
    """Backup all CSV files from data directory to backups/ with timestamp."""
    # Create backups directory if it doesn't exist
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_dir = backup_dir / timestamp
    timestamped_dir.mkdir(exist_ok=True)

    # Find all CSV files in data directory
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    # Copy each CSV file to backup
    for csv_file in csv_files:
        dest = timestamped_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        print(f"✓ Backed up: {csv_file.name} → backups/{timestamp}/{csv_file.name}")

    print(f"\nBackup complete! {len(csv_files)} file(s) backed up to backups/{timestamp}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup CSV files before trading")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing CSV files (default: current directory)"
    )
    args = parser.parse_args()

    backup_csvs(args.data_dir)
