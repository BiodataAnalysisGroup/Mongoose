#!/usr/bin/env python3
"""
Hyperparameter Tuning Pipeline for UnitedNet PerturbMap

This script runs systematic hyperparameter tuning experiments for the PerturbMap
technique, focusing initially on learning rate but extensible to other parameters.

Features:
- Tests multiple learning rate values
- Runs complete pipeline: training → SHAP analysis → feature analysis → validation
- Tracks experiment results and generates comparison reports
- Handles failed runs gracefully
- Creates organized output structure
- Generates summary reports

Usage:
    python hyperparameter_tuning.py --dataset_id KP2_1 --param learning_rate --values 0.0001 0.0005 0.001 0.005 0.01
    python hyperparameter_tuning.py --dataset_id KP2_1 --param train_epochs --values 10 15 20 25 30
"""

import argparse
import json
import subprocess
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import shutil
from typing import Dict, List, Any, Optional
import traceback
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hyperparameter_tuning.log')
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Systematic hyperparameter tuning for PerturbMap pipeline"""
    
    def __init__(self, 
                 dataset_id: str,
                 data_path: str = "../Data/UnitedNet/input_data",
                 base_output_dir: str = "../Mongoose_post_hoc",
                 genki_file: str = None,
                 deg_file: str = None):
        
        self.dataset_id = dataset_id
        self.data_path = Path(data_path)
        self.base_output_dir = Path(base_output_dir)
        self.genki_file = genki_file
        self.deg_file = deg_file
        
        # Create main experiment directory with timestamp
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_output_dir / f"experiment_{self.experiment_timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.experiment_results = []
        self.successful_runs = []
        self.failed_runs = []
        
        # Default configuration
        self.base_config = self._get_base_config()
        
        logger.info(f"Initialized HyperparameterTuner for dataset {dataset_id}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration for PerturbMap"""
        return {
            'train_batch_size': 8,
            'finetune_batch_size': 8,
            'train_epochs': 20,
            'finetune_epochs': 10,
            'lr': 0.001,
            'n_clusters': 3,
            'noise_level': [0, 0, 0],
            'technique': 'perturbmap'
        }
    
    def run_single_experiment(self, 
                            param_name: str, 
                            param_value: Any,
                            experiment_id: str) -> Dict[str, Any]:
        """Run a single hyperparameter experiment"""
        
        logger.info(f"Starting experiment {experiment_id}: {param_name}={param_value}")
        
        # Create experiment-specific directory
        exp_dir = self.experiment_dir / f"exp_{experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        
        experiment_result = {
            'experiment_id': experiment_id,
            'param_name': param_name,
            'param_value': param_value,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'model_path': None,
            'shap_dir': None,
            'analysis_dir': None,
            'validation_results': None,
            'error_message': None,
            'duration_minutes': None
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Train model with hyperparameter
            logger.info(f"Step 1: Training model for {experiment_id}")
            model_path = self._train_model(param_name, param_value, exp_dir)
            experiment_result['model_path'] = str(model_path)
            
            # Step 2: Calculate SHAP values
            logger.info(f"Step 2: Calculating SHAP for {experiment_id}")
            shap_dir = self._calculate_shap(model_path, exp_dir)
            experiment_result['shap_dir'] = str(shap_dir)
            
            # Step 3: Feature-feature analysis
            logger.info(f"Step 3: Feature analysis for {experiment_id}")
            analysis_dir = self._feature_analysis(shap_dir, exp_dir)
            experiment_result['analysis_dir'] = str(analysis_dir)
            
            # Step 4: Validation (if files provided)
            if self.genki_file and self.deg_file:
                logger.info(f"Step 4: Validation for {experiment_id}")
                validation_results = self._run_validation(analysis_dir, exp_dir)
                experiment_result['validation_results'] = validation_results
            
            # Calculate duration
            duration = (time.time() - start_time) / 60
            experiment_result['duration_minutes'] = round(duration, 2)
            experiment_result['status'] = 'completed'
            
            self.successful_runs.append(experiment_result)
            logger.info(f"Completed experiment {experiment_id} in {duration:.1f} minutes")
            
        except Exception as e:
            error_msg = f"Error in experiment {experiment_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            duration = (time.time() - start_time) / 60
            experiment_result['duration_minutes'] = round(duration, 2)
            experiment_result['status'] = 'failed'
            experiment_result['error_message'] = error_msg
            
            self.failed_runs.append(experiment_result)
        
        experiment_result['end_time'] = datetime.now().isoformat()
        return experiment_result
    
    def _train_model(self, param_name: str, param_value: Any, exp_dir: Path) -> Path:
        """Train model with specific hyperparameter"""
        
        # Create model output directory
        model_dir = exp_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        # Build training command
        cmd = [
            sys.executable, "uni_training.py",
            "--data_path", str(self.data_path),
            "--dataset_id", self.dataset_id,
            "--output_base", str(model_dir),
            "--timestamp"  # Add timestamp to avoid conflicts
        ]
        
        # Add hyperparameter-specific arguments
        if param_name == 'learning_rate' or param_name == 'lr':
            cmd.extend(["--lr", str(param_value)])
        elif param_name == 'train_epochs':
            cmd.extend(["--train_epochs", str(param_value)])
        elif param_name == 'finetune_epochs':
            cmd.extend(["--finetune_epochs", str(param_value)])
        elif param_name == 'train_batch_size':
            cmd.extend(["--train_batch_size", str(param_value)])
        elif param_name == 'finetune_batch_size':
            cmd.extend(["--finetune_batch_size", str(param_value)])
        elif param_name == 'n_clusters':
            cmd.extend(["--n_clusters", str(param_value)])
        
        # Run training
        logger.info(f"Running training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Find the created model file
        model_files = list(model_dir.glob("**/model_perturbmap_*.pkl"))
        if not model_files:
            raise RuntimeError("No model file found after training")
        
        return model_files[0]  # Return the first (should be only) model file
    
    def _calculate_shap(self, model_path: Path, exp_dir: Path) -> Path:
        """Calculate SHAP values for the trained model"""
        
        shap_dir = exp_dir / "shap_analysis"
        shap_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, "calculate_shap_values.py",
            "--model_path", str(model_path),
            "--dataset_id", self.dataset_id,
            "--data_path", str(self.data_path),
            "--output_dir", str(shap_dir)
        ]
        
        logger.info(f"Running SHAP command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            raise RuntimeError(f"SHAP calculation failed: {result.stderr}")
        
        # Find the created SHAP directory
        shap_dirs = [d for d in shap_dir.iterdir() if d.is_dir() and d.name.startswith("shap_analysis_")]
        if not shap_dirs:
            raise RuntimeError("No SHAP analysis directory found")
        
        return shap_dirs[0]
    
    def _feature_analysis(self, shap_dir: Path, exp_dir: Path) -> Path:
        """Run feature-to-feature analysis"""
        
        analysis_dir = exp_dir / "cross_modal_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, "feature_feature_analysis.py",
            "--dataset_id", self.dataset_id,
            "--shap_dir", str(shap_dir),
            "--data_path", str(self.data_path),
            "--output_dir", str(analysis_dir)
        ]
        
        logger.info(f"Running feature analysis command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            raise RuntimeError(f"Feature analysis failed: {result.stderr}")
        
        # Find the created analysis directory
        analysis_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith("cross_modal_analysis_")]
        if not analysis_dirs:
            raise RuntimeError("No cross-modal analysis directory found")
        
        return analysis_dirs[0]
    
    def _run_validation(self, analysis_dir: Path, exp_dir: Path) -> Dict[str, Any]:
        """Run validation using complete_perturbmap_analyzer.py"""
        
        validation_dir = exp_dir / "validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Find the feature importance CSV file
        feature_files = list(analysis_dir.glob("**/feature_feature_importance_3modalities.csv"))
        if not feature_files:
            raise RuntimeError("Feature importance CSV not found for validation")
        
        feature_file = feature_files[0]
        
        cmd = [
            sys.executable, "perturbmap_analyzer.py",
            "--shap_file", str(feature_file),
            "--genki_file", self.genki_file,
            "--deg_file", self.deg_file,
            "--output_dir", str(validation_dir)
        ]
        
        logger.info(f"Running validation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            raise RuntimeError(f"Validation failed: {result.stderr}")
        
        # Extract validation metrics from the results
        validation_results = self._extract_validation_metrics(validation_dir)
        return validation_results
    
    def _extract_validation_metrics(self, validation_dir: Path) -> Dict[str, Any]:
        """Extract key metrics from validation results"""
        
        # Look for analysis_data.json
        json_files = list(validation_dir.glob("**/analysis_data.json"))
        if not json_files:
            return {"error": "No validation results found"}
        
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            metrics = {
                'matched_genes': data['data_summary']['matched_genes'],
                'unique_molecules': data['data_summary']['unique_molecules'],
                'deg_coverage_percent': data['deg_overlap']['coverage_percent'],
                'genki_validation_percent': data['deg_overlap']['genki_percent'],
                'total_deg_matches': data['deg_overlap']['total_matches'],
                'shap_matches': data['deg_overlap']['shap_matches']
            }
            
            return metrics
            
        except Exception as e:
            return {"error": f"Error extracting metrics: {str(e)}"}
    
    def run_hyperparameter_sweep(self, 
                                param_name: str, 
                                param_values: List[Any]) -> List[Dict[str, Any]]:
        """Run complete hyperparameter sweep"""
        
        logger.info(f"Starting hyperparameter sweep: {param_name} with {len(param_values)} values")
        logger.info(f"Values to test: {param_values}")
        
        # Save experiment configuration
        experiment_config = {
            'dataset_id': self.dataset_id,
            'data_path': str(self.data_path),
            'experiment_timestamp': self.experiment_timestamp,
            'parameter_name': param_name,
            'parameter_values': param_values,
            'base_config': self.base_config,
            'genki_file': self.genki_file,
            'deg_file': self.deg_file
        }
        
        config_file = self.experiment_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(experiment_config, f, indent=2, default=str)
        
        # Run experiments
        for i, param_value in enumerate(param_values):
            experiment_id = f"{param_name}_{param_value}".replace('.', '_').replace('-', '_')
            
            logger.info(f"Running experiment {i+1}/{len(param_values)}: {experiment_id}")
            
            result = self.run_single_experiment(param_name, param_value, experiment_id)
            self.experiment_results.append(result)
            
            # Save intermediate results
            self._save_results()
        
        # Generate final report
        self._generate_final_report()
        
        logger.info(f"Hyperparameter sweep completed!")
        logger.info(f"Successful runs: {len(self.successful_runs)}")
        logger.info(f"Failed runs: {len(self.failed_runs)}")
        
        return self.experiment_results
    
    def _save_results(self):
        """Save current experiment results"""
        results_file = self.experiment_dir / "experiment_results.json"
        
        results_data = {
            'experiment_timestamp': self.experiment_timestamp,
            'dataset_id': self.dataset_id,
            'last_updated': datetime.now().isoformat(),
            'total_experiments': len(self.experiment_results),
            'successful_experiments': len(self.successful_runs),
            'failed_experiments': len(self.failed_runs),
            'results': self.experiment_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        
        logger.info("Generating final report...")
        
        # Create summary CSV
        if self.experiment_results:
            df_results = pd.DataFrame(self.experiment_results)
            csv_file = self.experiment_dir / "experiment_summary.csv"
            df_results.to_csv(csv_file, index=False)
            
            # Create markdown report
            report = self._create_markdown_report(df_results)
            report_file = self.experiment_dir / "experiment_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Final report saved to: {report_file}")
            logger.info(f"Summary CSV saved to: {csv_file}")
    
    def _create_markdown_report(self, df_results: pd.DataFrame) -> str:
        """Create markdown summary report"""
        
        successful_df = df_results[df_results['status'] == 'completed']
        failed_df = df_results[df_results['status'] == 'failed']
        
        report = f"""# Hyperparameter Tuning Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset**: {self.dataset_id}  
**Experiment ID**: {self.experiment_timestamp}

## Summary

- **Total Experiments**: {len(self.experiment_results)}
- **Successful**: {len(successful_df)}
- **Failed**: {len(failed_df)}
- **Success Rate**: {len(successful_df)/len(self.experiment_results)*100:.1f}%

## Parameter Analysis

"""
        
        if len(successful_df) > 0:
            param_name = self.experiment_results[0]['param_name']
            
            # Performance summary
            report += f"### {param_name.title()} Performance Summary\n\n"
            report += "| Value | Status | Duration (min) | Matched Genes | DEG Coverage % | GenKI Validation % |\n"
            report += "|-------|--------|----------------|---------------|----------------|-------------------|\n"
            
            for _, row in df_results.iterrows():
                status = row['status']
                duration = row.get('duration_minutes', 'N/A')
                
                if status == 'completed' and row.get('validation_results'):
                    val_res = row['validation_results']
                    matched_genes = val_res.get('matched_genes', 'N/A')
                    deg_coverage = val_res.get('deg_coverage_percent', 'N/A')
                    genki_validation = val_res.get('genki_validation_percent', 'N/A')
                    
                    if isinstance(deg_coverage, (int, float)):
                        deg_coverage = f"{deg_coverage:.1f}"
                    if isinstance(genki_validation, (int, float)):
                        genki_validation = f"{genki_validation:.1f}"
                else:
                    matched_genes = 'N/A'
                    deg_coverage = 'N/A'
                    genki_validation = 'N/A'
                
                report += f"| {row['param_value']} | {status} | {duration} | {matched_genes} | {deg_coverage} | {genki_validation} |\n"
            
            # Best performing configuration
            if len(successful_df) > 0 and 'validation_results' in successful_df.columns:
                best_idx = None
                best_score = -1
                
                for idx, row in successful_df.iterrows():
                    val_res = row.get('validation_results', {})
                    if isinstance(val_res, dict) and 'deg_coverage_percent' in val_res:
                        score = val_res['deg_coverage_percent']
                        if isinstance(score, (int, float)) and score > best_score:
                            best_score = score
                            best_idx = idx
                
                if best_idx is not None:
                    best_row = successful_df.loc[best_idx]
                    report += f"\n### Best Performing Configuration\n\n"
                    report += f"**{param_name}**: {best_row['param_value']}  \n"
                    report += f"**DEG Coverage**: {best_score:.1f}%  \n"
                    report += f"**Duration**: {best_row['duration_minutes']:.1f} minutes  \n"
        
        # Failed experiments
        if len(failed_df) > 0:
            report += "\n## Failed Experiments\n\n"
            for _, row in failed_df.iterrows():
                report += f"- **{row['param_value']}**: {row.get('error_message', 'Unknown error')}\n"
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        if len(successful_df) > 0:
            report += "Based on the results:\n\n"
            report += "1. All successful configurations can be used for production\n"
            report += "2. Consider the trade-off between performance and training time\n"
            report += "3. The best performing configuration shows optimal biological relevance\n"
        else:
            report += "No successful runs completed. Consider:\n\n"
            report += "1. Checking data paths and file availability\n"
            report += "2. Adjusting parameter ranges\n" 
            report += "3. Reviewing error messages above\n"
        
        report += f"\n---\n*Report generated by HyperparameterTuner*"
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning pipeline for UnitedNet PerturbMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test different learning rates
  python hyperparameter_tuning.py --dataset_id KP2_1 --param learning_rate --values 0.0001 0.0005 0.001 0.005 0.01
  
  # Test different epoch counts
  python hyperparameter_tuning.py --dataset_id KP2_1 --param train_epochs --values 10 15 20 25 30
  
  # Test with validation files
  python hyperparameter_tuning.py --dataset_id KP2_1 --param learning_rate --values 0.001 0.005 0.01 --genki_file genki.csv --deg_file degs.csv
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='Dataset identifier (e.g., KP2_1)')
    parser.add_argument('--param', type=str, required=True,
                       choices=['learning_rate', 'lr', 'train_epochs', 'finetune_epochs', 
                               'train_batch_size', 'finetune_batch_size', 'n_clusters'],
                       help='Parameter to tune')
    parser.add_argument('--values', nargs='+', required=True,
                       help='Parameter values to test')
    
    # Optional arguments
    parser.add_argument('--data_path', type=str, default='../Data/UnitedNet/input_data',
                       help='Path to input data directory')
    parser.add_argument('--output_dir', type=str, default='../Hyperparameter_Tuning',
                       help='Base output directory for experiments')
    parser.add_argument('--genki_file', type=str, default=None,
                       help='GenKI results file for validation (optional)')
    parser.add_argument('--deg_file', type=str, default=None,
                       help='DEG file for validation (optional)')
    parser.add_argument('--continue_on_failure', action='store_true',
                       help='Continue with remaining experiments if one fails')
    
    args = parser.parse_args()
    
    # Convert values to appropriate types
    param_values = []
    for value in args.values:
        try:
            # Try int first
            if '.' not in value:
                param_values.append(int(value))
            else:
                # Then float
                param_values.append(float(value))
        except ValueError:
            # Keep as string
            param_values.append(value)
    
    logger.info(f"Starting hyperparameter tuning for {args.param}")
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Values: {param_values}")
    
    # Create tuner
    tuner = HyperparameterTuner(
        dataset_id=args.dataset_id,
        data_path=args.data_path,
        base_output_dir=args.output_dir,
        genki_file=args.genki_file,
        deg_file=args.deg_file
    )
    
    try:
        # Run hyperparameter sweep
        results = tuner.run_hyperparameter_sweep(args.param, param_values)
        
        # Print summary
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("="*70)
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        print(f"\nResults saved to: {tuner.experiment_dir}")
        print("="*70)
        
        return 0 if len(successful) > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())