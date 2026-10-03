import sys
import argparse

from scripts.inspect_dataset import main as inspect_dataset_main
from scripts.prepare_dataset import prepare_dataset
from enhancement.train import train_enhancement
from enhancement.evaluate import evaluate_enhancement
from detection.train import train_detector
from detection.evaluate import evaluate_detector
from scripts.generate_comparison import generate_comparison
from robustness.evaluate import run_robustness_experiments

def main():
    parser = argparse.ArgumentParser(description="Low-Light Robust Perception End-to-End Pipeline")
    parser.add_argument("--inspect", action="store_true", help="Inspect raw ExDark dataset")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare train/val/test splits and YOLO labels")
    parser.add_argument("--train-enhancement", action="store_true", help="Train Zero-DCE++ low-light enhancer")
    parser.add_argument("--train-detector", action="store_true", help="Train YOLO object detector")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate baseline and enhanced detection models")
    parser.add_argument("--robustness", action="store_true", help="Run synthetic degradation robustness experiments")
    parser.add_argument("--quick-test", action="store_true", help="Run complete pipeline in fast test mode")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline end-to-end")

    args = parser.parse_args()

    # If no flags passed, default to --help or quick-test reminder
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    quick = args.quick_test

    if args.inspect or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 1: DATASET INSPECTION")
        print("=" * 60)
        inspect_dataset_main()

    if args.prepare_data or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 2: DATASET PREPARATION (COCO -> YOLO)")
        print("=" * 60)
        prepare_dataset(quick_test=quick)

    if args.train_enhancement or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 3: TRAIN ZERO-DCE++ LOW-LIGHT ENHANCER")
        print("=" * 60)
        train_enhancement(quick_test=quick)

    if args.evaluate or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 4: EVALUATE ENHANCEMENT MODEL & PREPARE ENHANCED TEST SET")
        print("=" * 60)
        evaluate_enhancement(quick_test=quick)

    if args.train_detector or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 5: TRAIN YOLO OBJECT DETECTOR")
        print("=" * 60)
        train_detector(quick_test=quick)

    if args.evaluate or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 6: EVALUATE BASELINE & ENHANCED OBJECT DETECTION")
        print("=" * 60)
        print("\nEvaluating Baseline (Original Low-Light + YOLO)...")
        evaluate_detector(is_enhanced=False)

        print("\nEvaluating Enhanced Pipeline (Zero-DCE++ + YOLO)...")
        evaluate_detector(is_enhanced=True)

        print("\nGenerating Controlled Comparison Table & Figures...")
        generate_comparison(quick_test=quick)

    if args.robustness or args.all or quick:
        print("\n" + "=" * 60)
        print("STEP 7: SYNTHETIC DEGRADATION ROBUSTNESS EXPERIMENTS")
        print("=" * 60)
        run_robustness_experiments(quick_test=quick)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
