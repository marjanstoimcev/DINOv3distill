#!/usr/bin/env python3
"""
Download TACO (Trash Annotations in Context) dataset from Roboflow Universe.

Usage:
    python download_roboflow_dataset.py --api-key YOUR_API_KEY

    # Or set environment variable:
    export ROBOFLOW_API_KEY=YOUR_API_KEY
    python download_roboflow_dataset.py

Get your API key from: https://app.roboflow.com/settings/api
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TACO dataset from Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_roboflow_dataset.py --api-key rf_xxxxxxxxxxxxx
    python download_roboflow_dataset.py -k rf_xxxxxxxxxxxxx

    # Using environment variable:
    export ROBOFLOW_API_KEY=rf_xxxxxxxxxxxxx
    python download_roboflow_dataset.py

Get your API key from:
    https://app.roboflow.com/settings/api
        """
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env variable)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/taco",
        help="Output directory (default: ./data/taco)"
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        default="yolov8",
        choices=["yolov8", "coco", "voc", "yolov5"],
        help="Dataset format (default: yolov8 - use yolov8 for segmentation)"
    )

    parser.add_argument(
        "--version", "-v",
        type=int,
        default=1,
        help="Dataset version (default: 1)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="mohamed-traore-2ekkp",
        help="Roboflow workspace name"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="taco-trash-annotations-in-context",
        help="Roboflow project name"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check API key
    if not args.api_key:
        print("Error: API key is required!")
        print("\nUsage:")
        print("  python download_roboflow_dataset.py --api-key YOUR_API_KEY")
        print("\nOr set environment variable:")
        print("  export ROBOFLOW_API_KEY=YOUR_API_KEY")
        print("\nGet your API key from:")
        print("  https://app.roboflow.com/settings/api")
        sys.exit(1)

    # Install roboflow if needed
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system("pip install roboflow -q")
        from roboflow import Roboflow

    print(f"\n{'='*60}")
    print("Downloading TACO Dataset from Roboflow")
    print(f"{'='*60}")
    print(f"Workspace: {args.workspace}")
    print(f"Project:   {args.project}")
    print(f"Output:    {args.output}")
    print(f"Format:    {args.format}")
    print(f"Version:   {args.version}")
    print()

    # Connect to Roboflow
    rf = Roboflow(api_key=args.api_key)

    # Load project
    print("Loading project from Roboflow Universe...")
    try:
        project = rf.workspace(args.workspace).project(args.project)
    except Exception as e:
        print(f"\nError loading project: {e}")
        print(f"\nMake sure the workspace/project exists:")
        print(f"  https://universe.roboflow.com/{args.workspace}/{args.project}")
        sys.exit(1)

    # Try to get the specified version
    try:
        version = project.version(args.version)
    except RuntimeError as e:
        print(f"\nVersion {args.version} not found.")
        print("\nTrying to list available versions...")
        try:
            versions = project.versions()
            if versions:
                print(f"Available versions: {[v.version for v in versions]}")
                latest = versions[-1]
                version_num = int(latest.version.split('/')[-1]) if hasattr(latest, 'version') else 1
                print(f"Using version: {version_num}")
                version = project.version(version_num)
            else:
                print("No versions available.")
                sys.exit(1)
        except Exception as e2:
            print(f"Error: {e2}")
            sys.exit(1)

    print("Downloading... (this may take a few minutes)")

    try:
        dataset = version.download(args.format, location=args.output, overwrite=True)
    except Exception as e:
        print(f"\nDownload error: {e}")
        print("\nTry downloading manually:")
        print(f"  1. Go to: https://universe.roboflow.com/{args.workspace}/{args.project}")
        print(f"  2. Click 'Download Dataset'")
        print(f"  3. Select '{args.format}' format")
        print(f"  4. Extract to: {args.output}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"{'='*60}")
    print(f"Location: {dataset.location}")

    # Show contents
    total_images = 0
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(args.output, split, 'images')
        if os.path.exists(split_dir):
            n_images = len([f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.png'))])
            print(f"  {split}: {n_images} images")
            total_images += n_images

    print(f"\nTotal: {total_images} images")
    print(f"\nDataset ready for training!")


if __name__ == "__main__":
    main()
