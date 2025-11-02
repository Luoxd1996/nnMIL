#!/usr/bin/env python3
"""
nnMIL Plan Experiment - Command line interface

Similar to nnUNetv2_plan_experiment, this script plans experiments by:
1. Analyzing feature files
2. Creating patient-level stratified data splits
3. Generating training configurations
4. Saving plan file

Usage:
    nnMIL_plan_experiment -d Dataset001_ebrains
"""
import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnMIL.preprocessing import ExperimentPlanner


def main():
    parser = argparse.ArgumentParser(
        description='Plan nnMIL experiment by analyzing dataset and generating configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plan experiment for Dataset001
    nnMIL_plan_experiment -d examples/Dataset001_ebrains
    
    # Plan with custom random seed
    nnMIL_plan_experiment -d examples/Dataset002_tcga_brca --seed 123
    
    # Plan and save to custom location
    nnMIL_plan_experiment -d examples/Dataset003_dss -o custom_plan.json
        """
    )
    
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                       help='Directory containing dataset.json and dataset.csv')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for plan file (default: dataset_dir/dataset_plan.json)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing plan file if it exists')
    
    args = parser.parse_args()
    
    # Check if plan file exists
    dataset_dir = args.dataset_dir
    if args.output is None:
        output_path = os.path.join(dataset_dir, 'dataset_plan.json')
    else:
        output_path = args.output
    
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Plan file already exists: {output_path}")
        print("Use --overwrite to overwrite existing plan")
        return
    
    # Run planning
    print(f"=" * 60)
    print(f"nnMIL Experiment Planning")
    print(f"=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output plan file: {output_path}")
    print(f"Random seed: {args.seed}")
    print(f"=" * 60)
    print()
    
    try:
        planner = ExperimentPlanner(dataset_dir, random_seed=args.seed)
        plan = planner.plan_experiment(output_path=output_path)
        
        print()
        print("=" * 60)
        print("Planning Summary")
        print("=" * 60)
        print(f"Feature dimension: {plan['feature_statistics']['feature_dimension']}")
        print(f"Recommended max_seq_length: {plan['feature_statistics']['recommended_max_seq_length']}")
        print(f"Batch size: {plan['training_configuration']['batch_size']}")
        print("=" * 60)
        print(f"\n✅ Planning complete! Plan saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during planning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

