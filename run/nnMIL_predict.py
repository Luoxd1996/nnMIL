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
from nnMIL.utilities.plan_loader import create_dataset_from_plan, get_config_from_plan, get_dataset_info_from_plan


def main():
    parser = argparse.ArgumentParser(
        description='nnMIL inference/prediction script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict using plan file (recommended)
    nnMIL_predict --plan_path examples/Dataset001_ebrains/dataset_plan.json \\
                  --checkpoint_path checkpoints/best_model.pth \\
                  --output_dir predictions/fold0
    
    # Predict for specific fold in 5-fold CV
    nnMIL_predict --plan_path examples/Dataset002_tcga_brca/dataset_plan.json \\
                  --checkpoint_path checkpoints/fold0/best_model.pth \\
                  --fold 0 \\
                  --output_dir predictions/fold0
        """
    )
    
    # Plan-based workflow (recommended)
    parser.add_argument('--plan_path', '-p', type=str, default=None,
                       help='Path to dataset_plan.json')
    
    # Legacy workflow (for backward compatibility)
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
    
    # Initialize inference engine
    engine = InferenceEngine(
        plan_path=args.plan_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    
    # Load dataset
    if args.plan_path and os.path.exists(args.plan_path):
        # Plan-based workflow
        test_dataset = create_dataset_from_plan(args.plan_path, split='test', fold=args.fold)
        config = get_config_from_plan(args.plan_path)
        dataset_info = get_dataset_info_from_plan(args.plan_path)
        
        # Override config with command-line arguments if provided
        kwargs = {
            'model_type': args.model_type or config.get('model_type', 'simple_mil'),
            'input_dim': args.input_dim or config.get('feature_dimension', 2560),
            'hidden_dim': args.hidden_dim or config.get('hidden_dim', 512),
            'dropout': args.dropout or config.get('dropout', 0.25),
            'batch_size': args.batch_size,
        }
    else:
        # Legacy workflow - user must provide dataset manually
        raise ValueError("Plan-based workflow is required. Please provide --plan_path.")
    
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

