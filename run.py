"""
Main Orchestration Script for Predictive Maintenance System

This script provides a unified interface to run different components of the system:
- Data preprocessing
- Model training
- Model evaluation
- API server
- Dashboard
- Monitoring
- Retraining

Usage:
    python run.py --all                   # Run complete pipeline
    python run.py --step preprocess       # Run preprocessing only
    python run.py --step train            # Run training only
    python run.py --step evaluate         # Run evaluation only
    python run.py --step api              # Start API server
    python run.py --step dashboard        # Start dashboard
    python run.py --step monitor          # Run monitoring
    python run.py --step retrain          # Run retraining
"""

import argparse
import sys
import subprocess
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrate the execution of different pipeline components.
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.root_dir = Path(__file__).parent
        self.src_dir = self.root_dir / 'src'
        self.app_dir = self.root_dir / 'app'
        self.data_dir = self.root_dir / 'data'

        logger.info("Pipeline Orchestrator initialized")

    def check_dependencies(self) -> bool:
        """
        Check if required directories and files exist.

        Returns:
            True if all dependencies are met
        """
        logger.info("Checking dependencies...")

        # Check directories
        required_dirs = [self.src_dir, self.app_dir, self.data_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Required directory not found: {dir_path}")
                return False

        # Check dataset
        dataset_path = self.data_dir / 'raw' / 'ai4i2020.csv'
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.error("Please download the AI4I 2020 dataset and place it in data/raw/")
            return False

        logger.info("All dependencies checked successfully")
        return True

    def run_command(self, command: List[str], description: str) -> bool:
        """
        Run a command and handle errors.

        Args:
            command: Command to run as list
            description: Description of the command

        Returns:
            True if successful
        """
        logger.info(f"{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"{'='*60}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True,
                cwd=self.root_dir
            )
            logger.info(f"{description} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"{description} failed with exit code {e.returncode}")
            return False

        except Exception as e:
            logger.error(f"Error running {description}: {str(e)}")
            return False

    def run_preprocess(self) -> bool:
        """Run data preprocessing."""
        return self.run_command(
            [sys.executable, str(self.src_dir / 'preprocess.py')],
            "Data Preprocessing"
        )

    def run_train(self) -> bool:
        """Run model training."""
        return self.run_command(
            [sys.executable, str(self.src_dir / 'train.py')],
            "Model Training"
        )

    def run_evaluate(self) -> bool:
        """Run model evaluation."""
        return self.run_command(
            [sys.executable, str(self.src_dir / 'evaluate.py')],
            "Model Evaluation"
        )

    def run_api(self) -> bool:
        """Start API server."""
        logger.info("Starting API server...")
        logger.info("API will be available at: http://localhost:8000")
        logger.info("API documentation: http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the server")

        return self.run_command(
            [sys.executable, str(self.app_dir / 'app.py')],
            "API Server"
        )

    def run_dashboard(self) -> bool:
        """Start Streamlit dashboard."""
        logger.info("Starting Streamlit dashboard...")
        logger.info("Dashboard will be available at: http://localhost:8501")
        logger.info("Press Ctrl+C to stop the dashboard")

        return self.run_command(
            ['streamlit', 'run', str(self.app_dir / 'dashboard.py')],
            "Streamlit Dashboard"
        )

    def run_monitor(self) -> bool:
        """Run monitoring system."""
        return self.run_command(
            [sys.executable, str(self.src_dir / 'monitor.py')],
            "Monitoring System"
        )

    def run_retrain(self) -> bool:
        """Run retraining pipeline."""
        return self.run_command(
            [sys.executable, str(self.src_dir / 'retrain.py')],
            "Retraining Pipeline"
        )

    def run_test_api(self) -> bool:
        """Run API tests."""
        return self.run_command(
            [sys.executable, 'test_api.py'],
            "API Tests"
        )

   

    def run_all(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if all steps successful
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*60)

        steps = [
            ("Preprocessing", self.run_preprocess),
            ("Training", self.run_train),
            ("Evaluation", self.run_evaluate),
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*60}\n")

            if not step_func():
                logger.error(f"Pipeline failed at step: {step_name}")
                return False

        logger.info("\n" + "="*60)
        logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Start API: python run.py --step api")
        logger.info("  2. Start Dashboard: python run.py --step dashboard")
        logger.info("  3. Test API: python run.py --step test")
        logger.info("  4. Test API: python run.py --step test")
        return True

    def run_pipeline(self, step: str = None, run_all: bool = False) -> bool:
        """
        Run specified pipeline step(s).

        Args:
            step: Specific step to run
            run_all: Run complete pipeline

        Returns:
            True if successful
        """
        # Check dependencies first
        if not self.check_dependencies():
            return False

        if run_all:
            return self.run_all()

        # Map steps to functions
        step_map = {
            'preprocess': self.run_preprocess,
            'train': self.run_train,
            'evaluate': self.run_evaluate,
            'api': self.run_api,
            'dashboard': self.run_dashboard,
            'monitor': self.run_monitor,
            'retrain': self.run_retrain,
            'test': self.run_test_api,
        
        }

        if step not in step_map:
            logger.error(f"Invalid step: {step}")
            logger.error(f"Valid steps: {', '.join(step_map.keys())}")
            return False

        return step_map[step]()


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Predictive Maintenance System - Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run.py --all

  # Run specific steps
  python run.py --step preprocess
  python run.py --step train
  python run.py --step evaluate

  # Start services
  python run.py --step api
  python run.py --step dashboard
 

  # Run monitoring and retraining
  python run.py --step monitor
  python run.py --step retrain

  # Test API
  python run.py --step test
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run the complete pipeline (preprocess → train → evaluate)'
    )

    parser.add_argument(
        '--step',
        type=str,
        choices=['preprocess', 'train', 'evaluate', 'api', 'dashboard', 'monitor', 'retrain', 'test'],
        help='Run a specific pipeline step'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║            PREDICTIVE MAINTENANCE SYSTEM                  ║
    ║                                                           ║
    ║         End-to-End ML Pipeline Orchestrator               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """
    Main execution function.
    """
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.all and not args.step:
        logger.error("Please specify either --all or --step")
        logger.error("Run 'python run.py --help' for usage information")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()

    # Run pipeline
    try:
        success = orchestrator.run_pipeline(
            step=args.step,
            run_all=args.all
        )

        if success:
            logger.info("\nExecution completed successfully!")
            sys.exit(0)
        else:
            logger.error("\nExecution failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n\nExecution interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
