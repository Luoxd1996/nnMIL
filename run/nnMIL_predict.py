#!/usr/bin/env python3
"""
nnMIL Predict - Command line interface

Similar to nnUNetv2_predict, this script provides unified inference interface.

Usage:
    nnMIL_predict -i Dataset001_ebrains -m simple_mil -f fold0
    nnMIL_predict --plan_path examples/Dataset001_ebrains/dataset_plan.json --checkpoint_path checkpoints/best_model.pth
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from nnMIL.inference import InferenceEngine
from nnMIL.utilities.plan_loader import (
    create_dataset_from_plan, 
    get_config_from_plan, 
    get_dataset_info_from_plan,
    create_dataset_from_features_dir,
    find_plan_file
)


def main():
    parser = argparse.ArgumentParser(
        description='nnMIL inference/prediction script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # nnUNet-style: predict using task_id (recommended, automatically finds plan)
    nnMIL_predict -t Dataset001_ebrains \\
                  -c checkpoints/best_model.pth \\
                  -i /path/to/features \\
                  -o predictions/inference
    
    # Predict using plan file path (alternative)
    nnMIL_predict --plan_path examples/Dataset001_ebrains/dataset_plan.json \\
                  --checkpoint_path checkpoints/best_model.pth \\
                  --output_dir predictions/fold0
    
    # Predict for specific fold in 5-fold CV
    nnMIL_predict -t Dataset002_tcga_brca \\
                  -c checkpoints/fold0/best_model.pth \\
                  -f 0 \\
                  -o predictions/fold0
    
    # Predict from feature directory without task_id/plan (uses checkpoint config)
    nnMIL_predict --checkpoint_path checkpoints/best_model.pth \\
                  --input_dir /path/to/features \\
                  --output_dir predictions/inference \\
                  --model_type simple_mil --input_dim 2560
        """
    )
    
    # nnUNet-style: task_id (automatically finds plan file)
    parser.add_argument('--task_id', '-t', type=str, default=None,
                       help='Task ID (e.g., Dataset001_ebrains). Automatically finds plan file in examples/{task_id}/')
    
    # Alternative: direct plan file path
    parser.add_argument('--plan_path', '-p', type=str, default=None,
                       help='Path to dataset_plan.json (alternative to --task_id)')
    
    # Input options: either plan-based (from plan slide_info) or feature directory (nnUNet-style)
    parser.add_argument('--input_dir', '-i', type=str, default=None,
                       help='Input directory containing feature .h5 files (nnUNet-style, scans directory automatically)')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for predictions')
    
    # Optional arguments
    parser.add_argument('--fold', '-f', type=int, default=None,
                       help='Fold number for 5-fold CV (0-4). If None, use official split.')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (Note: For test/val, batch_size is forced to 1 due to variable-length sequences)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If None, auto-detect.')
    
    # Model override arguments (if not using plan file)
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type (overrides plan if specified)')
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (overrides plan if specified)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (overrides plan if specified)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (overrides plan if specified)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve plan_path: if task_id is provided, automatically find plan file
    plan_path = args.plan_path
    if args.task_id and not plan_path:
        try:
            plan_path = find_plan_file(args.task_id)
            print(f"📋 Found plan file for task '{args.task_id}': {plan_path}")
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("💡 You can still use --input_dir with --checkpoint_path for inference without plan")
            plan_path = None
    
    # Initialize inference engine
    engine = InferenceEngine(
        plan_path=plan_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    
    # Load dataset
    if args.input_dir and os.path.exists(args.input_dir):
        # nnUNet-style workflow: predict from feature directory
        print(f"📁 Using feature directory: {args.input_dir}")
        test_dataset = create_dataset_from_features_dir(
            features_dir=args.input_dir,
            plan_path=plan_path  # Optional: use plan for config only
        )
        
        # Get config from plan if provided, otherwise use defaults
        if plan_path and os.path.exists(plan_path):
            config = get_config_from_plan(plan_path)
            dataset_info = get_dataset_info_from_plan(plan_path)
        else:
            # Default config if no plan provided
            config = {}
            dataset_info = {'task_type': 'classification'}
            print("⚠️  No plan file provided, using default configuration")
        
        # Try to infer model_type from checkpoint path if not provided
        inferred_model_type = None
        if not args.model_type and args.checkpoint_path:
            # Extract from checkpoint path: best_{model_type}.pth or .../model_type/...
            checkpoint_basename = os.path.basename(args.checkpoint_path)
            if checkpoint_basename.startswith('best_') and checkpoint_basename.endswith('.pth'):
                # Format: best_{model_type}.pth
                inferred_model_type = checkpoint_basename[5:-4]  # Remove 'best_' and '.pth'
            else:
                # Try to extract from directory path: .../model_type/...
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                parts = checkpoint_dir.split(os.sep)
                # Look for common model types in path
                for part in reversed(parts):
                    if part in ['simple_mil', 'ab_mil', 'ds_mil', 'trans_mil', 'wikg_mil']:
                        inferred_model_type = part
                        break
        
        # Override config with command-line arguments if provided
        kwargs = {
            'model_type': args.model_type or inferred_model_type or config.get('model_type', 'simple_mil'),
            'input_dim': args.input_dim or config.get('feature_dimension', 2560),
            'hidden_dim': args.hidden_dim or config.get('hidden_dim', 512),
            'dropout': args.dropout or config.get('dropout', 0.25),
            'batch_size': args.batch_size,
        }
        
        # Log model type inference
        if inferred_model_type and not args.model_type:
            print(f"ℹ️  Inferred model_type='{inferred_model_type}' from checkpoint path")
            
    elif plan_path and os.path.exists(plan_path):
        # Plan-based workflow (from plan slide_info)
        print(f"📋 Using plan file: {plan_path}")
        test_dataset = create_dataset_from_plan(plan_path, split='test', fold=args.fold)
        config = get_config_from_plan(plan_path)
        dataset_info = get_dataset_info_from_plan(plan_path)
        
        # Try to infer model_type from checkpoint path if not provided
        inferred_model_type = None
        if not args.model_type and args.checkpoint_path:
            # Extract from checkpoint path: best_{model_type}.pth or .../model_type/...
            checkpoint_basename = os.path.basename(args.checkpoint_path)
            if checkpoint_basename.startswith('best_') and checkpoint_basename.endswith('.pth'):
                # Format: best_{model_type}.pth
                inferred_model_type = checkpoint_basename[5:-4]  # Remove 'best_' and '.pth'
            else:
                # Try to extract from directory path: .../model_type/...
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                parts = checkpoint_dir.split(os.sep)
                # Look for common model types in path
                for part in reversed(parts):
                    if part in ['simple_mil', 'ab_mil', 'ds_mil', 'trans_mil', 'wikg_mil']:
                        inferred_model_type = part
                        break
        
        # Override config with command-line arguments if provided
        kwargs = {
            'model_type': args.model_type or inferred_model_type or config.get('model_type', 'simple_mil'),
            'input_dim': args.input_dim or config.get('feature_dimension', 2560),
            'hidden_dim': args.hidden_dim or config.get('hidden_dim', 512),
            'dropout': args.dropout or config.get('dropout', 0.25),
            'batch_size': args.batch_size,
        }
        
        # Log model type inference
        if inferred_model_type and not args.model_type:
            print(f"ℹ️  Inferred model_type='{inferred_model_type}' from checkpoint path")
    else:
        # No valid input source
        raise ValueError(
            "Either --task_id/-t (nnUNet-style), --plan_path, or --input_dir must be provided.\n"
            "  - Use -t DatasetXXX for automatic plan lookup (recommended)\n"
            "  - Use --plan_path to specify plan file directly\n"
            "  - Use --input_dir with --checkpoint_path for inference from feature directory"
        )
    
    # Run prediction
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    results = engine.predict(
        test_dataset=test_dataset,
        save_dir=args.output_dir,
        logger=logger,
        **kwargs
    )
    
    print(f"✅ Inference completed. Results saved to {args.output_dir}")
    return results


if __name__ == "__main__":
    main()

