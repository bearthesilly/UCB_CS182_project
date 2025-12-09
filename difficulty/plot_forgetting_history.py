#!/usr/bin/env python3
"""
Script to plot forgetting history from multiple JSON files.
Graphs all metrics from JSON files with a legend based on the "name" parameter.
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_forgetting_histories(json_files: List[str], output_file: str = None):
    """
    Plot forgetting history data from multiple JSON files.

    Args:
        json_files: List of paths to JSON files
        output_file: Optional path to save the figure
    """
    data_list = []

    # Load all JSON files
    for filepath in json_files:
        try:
            data = load_json_file(filepath)
            data['filepath'] = filepath
            data_list.append(data)
            print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
            continue

    if not data_list:
        print("No valid JSON files loaded. Exiting.", file=sys.stderr)
        return

    # Identify all metric keys (excluding 'name', 'steps', 'filepath', and non-plottable metrics)
    all_metric_keys = set()
    excluded_keys = {'name', 'steps', 'filepath', 'mix_percentage', 'selection_method'}
    for data in data_list:
        for key in data.keys():
            if key not in excluded_keys:
                all_metric_keys.add(key)

    metric_keys = sorted(all_metric_keys)

    if not metric_keys:
        print("No metric data found in JSON files.", file=sys.stderr)
        return

    # Create subplots side by side with different widths
    num_metrics = len(metric_keys)

    if num_metrics == 1:
        # Single metric - use regular subplot
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    elif num_metrics == 2:
        # Two metrics - side by side with 40% and 60% width
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.4, 0.6], wspace=0.3)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    else:
        # More than 2 metrics - fall back to stacked layout
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        if not isinstance(axes, list):
            axes = [axes]

    # Plot each metric
    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        has_data = False

        for data in data_list:
            if metric_key not in data:
                continue

            steps = data.get('steps', [])
            metric_values = data.get(metric_key, [])

            # Use 'name' field for legend, fallback to filename if name is empty
            label = data.get('name', '')
            if not label or label.strip() == '':
                label = Path(data['filepath']).stem

            # Print debugging information
            print(f"\n{label} ({metric_key}):")
            print(f"  Steps length: {len(steps)}")
            print(f"  Metric values length: {len(metric_values)}")

            # Check for length mismatch
            if len(steps) != len(metric_values):
                print(f"WARNING: Length mismatch for {label} - {metric_key}: steps={len(steps)}, values={len(metric_values)}")
                print(f"  Skipping this series due to mismatch.")
                continue

            ax.plot(steps, metric_values, marker='o', label=label, linewidth=2, markersize=4)
            has_data = True

        # Format axis labels
        ax.set_xlabel('Steps', fontsize=16)

        # Determine dataset name based on key
        if 'hotpot' in metric_key.lower():
            dataset_name = 'HotPotQA'
        else:
            dataset_name = 'MATH'

        # Set y-axis label
        y_label = f'{dataset_name} Validation Loss'
        ax.set_ylabel(y_label, fontsize=16)

        # Set title
        title_text = f'{dataset_name} Validation Loss Over Training Steps'
        ax.set_title(title_text, fontsize=14, fontweight='bold')

        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Show legend only on the first (top) graph with larger font
        if has_data and idx == 0:
            ax.legend(loc='best', fontsize=14)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_file}")
    else:
        # Try to show, but save to default file if display not available
        try:
            plt.show(block=True)
        except Exception as e:
            default_output = 'forgetting_history_plot.png'
            plt.savefig(default_output, dpi=300, bbox_inches='tight')
            print(f"\nDisplay not available. Figure saved to: {default_output}")
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot forgetting history from multiple JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot multiple JSON files
  python plot_forgetting_history.py file1.json file2.json file3.json

  # Plot with wildcard pattern
  python plot_forgetting_history.py drive/MyDrive/forgetting_history_*.json

  # Save to output file
  python plot_forgetting_history.py -o output.png file1.json file2.json
        """
    )

    parser.add_argument('json_files', nargs='+', help='JSON files to plot')
    parser.add_argument('-o', '--output', help='Output file path (e.g., output.png)')

    args = parser.parse_args()

    # Expand any glob patterns in the file list
    expanded_files = []
    for pattern in args.json_files:
        path = Path(pattern)
        if '*' in pattern or '?' in pattern:
            # Handle glob patterns
            if path.is_absolute():
                matches = list(Path('/').glob(str(path).lstrip('/')))
            else:
                matches = list(Path('.').glob(pattern))
            expanded_files.extend([str(m) for m in matches])
        else:
            expanded_files.append(pattern)

    if not expanded_files:
        print("No files found matching the given patterns.", file=sys.stderr)
        return 1

    plot_forgetting_histories(expanded_files, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
