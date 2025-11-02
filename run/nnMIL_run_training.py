#!/usr/bin/env python3
"""
nnMIL Run Training - Unified command-line interface for training

Similar to nnUNetv2_train, this script provides a unified training interface:
- Automatic task type detection from plan file
- Automatic trainer selection
- Automatic path resolution
- Support for all four training tasks (classification, regression, survival, survival_porpoise)

Usage:
    nnMIL_run_training Dataset001_ebrains simple_mil 0
    nnMIL_run_training Dataset001_ebrains simple_mil all
    nnMIL_run_training Dataset001_ebrains simple_mil 0 --batch_size 64
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnMIL.utilities.plan_loader import load_plan


def find_plan_file(dataset_id: str) -> str:
    """
    Find plan file for a given dataset ID.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'Dataset001_ebrains')
    
    Returns:
        Path to dataset_plan.json
    
    Search order:
    1. examples/{dataset_id}/dataset_plan.json
    2. {dataset_id}/dataset_plan.json (if dataset_id is already a path)
    """
    # Check if dataset_id is already a full path
    if os.path.isabs(dataset_id) or os.path.exists(dataset_id):
        plan_path = os.path.join(dataset_id, 'dataset_plan.json') if os.path.isdir(dataset_id) else dataset_id
        if os.path.exists(plan_path):
            return plan_path
    
    # Try examples directory
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    plan_path = os.path.join(examples_dir, dataset_id, 'dataset_plan.json')
    if os.path.exists(plan_path):
        return plan_path
    
    # Try current directory
    plan_path = os.path.join(dataset_id, 'dataset_plan.json')
    if os.path.exists(plan_path):
        return plan_path
    
    raise FileNotFoundError(
        f"Could not find dataset_plan.json for dataset '{dataset_id}'. "
        f"Tried: {plan_path}"
    )


def select_trainer(plan_path: str, model_type: str, fold: int = None, **kwargs):
    """
    Automatically select the appropriate trainer based on plan file.
    
    Args:
        plan_path: Path to dataset_plan.json
        model_type: Model architecture
        fold: Cross-validation fold (None for official_split)
        **kwargs: Additional arguments
    
    Returns:
        Trainer instance
    """
    plan = load_plan(plan_path)
    # Plan file structure: task_type is at top level or in dataset_info
    if 'dataset_info' in plan:
        task_type = plan['dataset_info'].get('task_type')
        if not task_type:
            # Fallback: check top level
            task_type = plan.get('task_type')
    else:
        task_type = plan.get('task_type')
    
    if not task_type:
        raise ValueError(f"Could not find task_type in plan file: {plan_path}")
    
    config = plan.get('training_configuration', {})
    
    # Import trainers (lazy import to avoid circular dependencies)
    try:
        from nnMIL.training.trainers import (
            ClassificationTrainer,
            RegressionTrainer,
            SurvivalTrainer,
            SurvivalPorpoiseTrainer
        )
    except ImportError as e:
        raise ImportError(f"Failed to import trainers: {e}. Make sure all trainer files exist.")
    
    if task_type == 'classification':
        return ClassificationTrainer(plan_path, model_type, fold, **kwargs)
    
    elif task_type == 'regression':
        return RegressionTrainer(plan_path, model_type, fold, **kwargs)
    
    elif task_type == 'survival':
        # For survival tasks, need to distinguish between two trainers
        # Method 1: Check trainer_type in plan (if exists)
        trainer_type = config.get('trainer_type')
        if trainer_type == 'survival_porpoise':
            return SurvivalPorpoiseTrainer(plan_path, model_type, fold, **kwargs)
        elif trainer_type == 'survival':
            return SurvivalTrainer(plan_path, model_type, fold, **kwargs)
        
        # Method 2: Auto-infer from configuration
        batch_size = config.get('batch_size', 32)
        survival_loss = config.get('survival_loss', 'cox')
        
        if batch_size == 1 and survival_loss == 'nllsurv':
            return SurvivalPorpoiseTrainer(plan_path, model_type, fold, **kwargs)
        else:
            return SurvivalTrainer(plan_path, model_type, fold, **kwargs)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def determine_folds_to_run(plan_path: str, fold_arg: str) -> list:
    """
    Determine which folds to run based on evaluation setting and fold argument.
    
    Args:
        plan_path: Path to dataset_plan.json
        fold_arg: 'all' or a specific fold number (0-4) or 'official_split'
    
    Returns:
        List of folds to run
    """
    plan = load_plan(plan_path)
    # Plan file structure: evaluation_setting is at top level or in dataset_info
    if 'dataset_info' in plan:
        evaluation_setting = plan['dataset_info'].get('evaluation_setting', 'official_split')
    else:
        evaluation_setting = plan.get('evaluation_setting', 'official_split')
    
    # Normalize evaluation_setting: handle both '5fold' and '5_fold'
    if evaluation_setting in ['5fold', '5_fold']:
        evaluation_setting = '5fold'
    
    # Handle fold_arg: can be 'all', 'official_split', or a number
    if fold_arg == 'official_split' or fold_arg == 'None':
        fold_arg = None
    
    if evaluation_setting == '5fold':
        if fold_arg == 'all':
            return list(range(5))
        elif fold_arg is None:
            # Default to fold 0 for 5fold
            return [0]
        else:
            fold_num = int(fold_arg)
            if fold_num < 0 or fold_num >= 5:
                raise ValueError(f"Fold must be between 0 and 4, got {fold_num}")
            return [fold_num]
    else:  # official_split
        return [None]  # None means official_split (no fold)


def main():
    parser = argparse.ArgumentParser(
        description='nnMIL unified training interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on fold 0
    nnMIL_run_training Dataset001_ebrains simple_mil 0
    
    # Train all folds
    nnMIL_run_training Dataset001_ebrains simple_mil all
    
    # Override batch size
    nnMIL_run_training Dataset001_ebrains simple_mil 0 --batch_size 64
    
    # Evaluation only
    nnMIL_run_training Dataset001_ebrains simple_mil 0 --eval_only --resume checkpoints/best_model.pth
        """
    )
    
    # Required arguments (nnUNet style)
    parser.add_argument('dataset_id', type=str,
                       help='Dataset ID (e.g., Dataset001_ebrains) or path to dataset directory')
    parser.add_argument('model_type', type=str,
                       help='Model architecture (e.g., simple_mil, ab_mil)')
    parser.add_argument('fold', type=str,
                       help='Fold number (0-4) or "all" for all folds')
    
    # Optional training overrides
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides plan)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides plan)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides plan)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides plan)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation, skip training (requires --resume)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Custom save directory (overrides auto-generated path)')
    
    args = parser.parse_args()
    
    # Find plan file
    try:
        plan_path = find_plan_file(args.dataset_id)
        print(f"Found plan file: {plan_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Determine folds to run
    try:
        folds_to_run = determine_folds_to_run(plan_path, args.fold)
        print(f"Will run folds: {folds_to_run}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Prepare kwargs for trainer
    trainer_kwargs = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'seed': args.seed,
        'save_dir': args.save_dir,
        'resume': args.resume,
        'eval_only': args.eval_only,
    }
    # Remove None values
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}
    
    # Run training for each fold
    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"Training fold: {fold if fold is not None else 'official_split'}")
        print(f"{'='*60}\n")
        
        # CRITICAL: Reset PyTorch state before each fold
        # This ensures no state leaks between folds
        import torch
        torch.set_grad_enabled(True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Select and initialize trainer
            trainer = select_trainer(plan_path, args.model_type, fold, **trainer_kwargs)
            
            # Run training (or evaluation only)
            if args.eval_only:
                if args.resume is None:
                    print("Error: --eval_only requires --resume")
                    sys.exit(1)
                trainer.load_checkpoint(args.resume)
                # Only evaluate test if test split exists
                if trainer.test_loader is not None and len(trainer.test_loader.dataset) > 0:
                    trainer.evaluate('test')
                else:
                    print("  Skipping test evaluation: No test split available")
            else:
                trainer.train()
                # Only evaluate test if test split exists
                if trainer.test_loader is not None and len(trainer.test_loader.dataset) > 0:
                    trainer.evaluate('test')
                else:
                    print("  Skipping test evaluation: No test split available")
            
            # Clean up after each fold
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during training fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()

