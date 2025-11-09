"""Utility functions for convergence analysis scripts."""

import os


def get_output_directory(default_dir):
    """Prompt user for output directory path.

    Args:
        default_dir: Default directory path if user doesn't provide one

    Returns:
        Absolute path to the output directory
    """
    print("\nOutput Configuration")
    print("-" * 70)
    user_input = input(f"Enter output directory path (default: {default_dir}): ").strip()

    output_dir = user_input if user_input else default_dir

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    return output_dir
