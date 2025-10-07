#!/usr/bin/env python3
"""Utility script for managing window counts across all categories."""

import sys
import os
import argparse
import json
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from musedfm.data.benchmark import Benchmark


def cleanup_window_counts(base_path: str = "/workspace/data/fm_eval_nested"):
    """Clean up old window count JSON files."""
    
    # Remove old format files (without parameters)
    old_pattern = os.path.join(base_path, "*_window_counts.json")
    for file_path in glob.glob(old_pattern):
        os.remove(file_path)
        print(f"Removed old format: {os.path.basename(file_path)}")
    
    # Remove from compressed tar directory too
    compressed_tar_path = "/workspace/data/fm_eval_compressed_tar"
    if os.path.exists(compressed_tar_path):
        old_pattern = os.path.join(compressed_tar_path, "*_window_counts.json")
        for file_path in glob.glob(old_pattern):
            os.remove(file_path)
            print(f"Removed old format from compressed tar: {os.path.basename(file_path)}")


def save_all_window_counts(base_path: str = "/workspace/data/fm_eval_nested", cleanup_first: bool = False, load_cached: bool = False, history_length: int = 30, forecast_horizon: int = 1, stride: int = 1):
    """Save window counts for all categories."""
    if cleanup_first:
        cleanup_window_counts(base_path)
    
    benchmark = Benchmark(base_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached)
    generated_files = []
    
    for category in benchmark:
        category.save_window_counts(base_path)
        
        # Get the filename that was generated
        filename = f"{category.category}_window_counts_h{category.history_length}_f{category.forecast_horizon}_s{category.stride}.json"
        json_path = os.path.join(base_path, filename)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                window_counts = json.load(f)
            generated_files.append((filename, window_counts))
    
    # Print summary of generated files
    print(f"\n=== Generated Window Count Files ===")
    total_datasets = 0
    total_windows = 0
    
    for filename, window_counts in generated_files:
        category_total = sum(window_counts.values())
        print(f"\n{filename}:")
        print(f"  Datasets: {len(window_counts)}")
        print(f"  Total windows: {category_total:,}")
        
        # Show all datasets by window count
        sorted_counts = sorted(window_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  All datasets:")
        for dataset_name, count in sorted_counts:
            print(f"    {dataset_name}: {count:,} windows")
        
        total_datasets += len(window_counts)
        total_windows += category_total
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total files generated: {len(generated_files)}")
    print(f"Total datasets: {total_datasets}")
    print(f"Total estimated windows: {total_windows:,}")


def load_window_counts_summary(base_path: str = "/workspace/data/fm_eval_nested", load_cached: bool = False, history_length: int = 30, forecast_horizon: int = 1, stride: int = 1):
    """Load and display summary of all window counts."""
    benchmark = Benchmark(base_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached)
    
    total_datasets = 0
    total_windows = 0
    
    for category in benchmark:
        window_counts = category.get_window_counts(use_cached=True)
        if window_counts:
            category_total = sum(window_counts.values())
            print(f"{category.category}: {len(window_counts)} datasets, {category_total:,} windows")
            total_datasets += len(window_counts)
            total_windows += category_total
    
    print(f"Total: {total_datasets} datasets, {total_windows:,} windows")


def main():
    parser = argparse.ArgumentParser(description="Manage window counts for MUSED-FM datasets")
    parser.add_argument("--action", choices=["save", "load", "both", "cleanup"], default="both")
    parser.add_argument("--base-path", default="/workspace/data/fm_eval_nested")
    parser.add_argument("--cleanup-first", action="store_true", help="Clean up old window count files before saving")
    parser.add_argument("--load-cached", action="store_true", help="Load window counts from cached JSON files instead of generating")
    parser.add_argument("--history-length", type=int, default=512, help="History length for windows (default: 512)")
    parser.add_argument("--forecast-horizon", type=int, default=128, help="Forecast horizon for windows (default: 128)")
    parser.add_argument("--stride", type=int, default=256, help="Stride between windows (default: 256)")
    
    args = parser.parse_args()
    
    if args.action == "cleanup":
        cleanup_window_counts(args.base_path)
    elif args.action in ["save", "both"]:
        save_all_window_counts(args.base_path, args.cleanup_first, args.load_cached, args.history_length, args.forecast_horizon, args.stride)
    
    if args.action in ["load", "both"]:
        load_window_counts_summary(args.base_path, args.load_cached, args.history_length, args.forecast_horizon, args.stride)


if __name__ == "__main__":
    main()