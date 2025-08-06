#!/usr/bin/env python3
"""
Test script for comparing loss-based view selection vs baseline random sampling
Usage: python test_loss_judge.py --dataset <path_to_dataset>
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path

def run_training(dataset_path, output_dir, use_loss_judge=False, iterations=2000, num_selected_views=16, selection_interval=16):
    """Run training with specified parameters"""
    
    cmd = [
        "python", "train.py",
        "-s", dataset_path,
        "-m", output_dir,
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),  # Test at the end only
        "--save_iterations", str(iterations)   # Save at the end only
    ]
    
    if use_loss_judge:
        cmd.extend([
            "--loss_judge",
            "--num_selected_views", str(num_selected_views),
            "--selection_interval", str(selection_interval)
        ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        training_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None, training_time
            
        return result.stdout, training_time
    except Exception as e:
        print(f"Error running training: {e}")
        return None, time.time() - start_time

def extract_psnr_from_output(output_text):
    """Extract final PSNR from training output"""
    lines = output_text.split('\\n')
    for line in reversed(lines):  # Search from end
        if 'Evaluating test: L1' in line and 'PSNR' in line:
            # Parse line like: "[ITER 2000] Evaluating test: L1 0.123456 PSNR 28.123456"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'PSNR' and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        continue
    return None

def run_evaluation(model_path):
    """Run evaluation on trained model to get PSNR"""
    cmd = ["python", "render.py", "-m", model_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            print(f"Render failed: {result.stderr}")
            return None
            
        # Run metrics
        cmd = ["python", "metrics.py", "-m", model_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            print(f"Metrics failed: {result.stderr}")
            return None
            
        return result.stdout
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test loss-based view selection")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of training iterations")
    parser.add_argument("--num_selected_views", type=int, default=16, help="Number of views to select")
    parser.add_argument("--selection_interval", type=int, default=16, help="Selection interval")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for each method")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset path does not exist: {args.dataset}")
        sys.exit(1)
    
    dataset_name = Path(args.dataset).name
    results = {
        "dataset": dataset_name,
        "iterations": args.iterations,
        "num_selected_views": args.num_selected_views,
        "selection_interval": args.selection_interval,
        "baseline": [],
        "loss_judge": []
    }
    
    print(f"Testing on dataset: {dataset_name}")
    print(f"Iterations: {args.iterations}")
    print(f"Number of runs per method: {args.runs}")
    print("="*60)
    
    # Test baseline (random sampling)
    print("\\nTesting BASELINE (Random Sampling)...")
    for run in range(args.runs):
        print(f"  Run {run + 1}/{args.runs}")
        
        output_dir = f"./test_results/baseline_{dataset_name}_run{run}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_text, training_time = run_training(
            args.dataset, output_dir, use_loss_judge=False, iterations=args.iterations
        )
        
        if output_text:
            psnr = extract_psnr_from_output(output_text)
            results["baseline"].append({
                "run": run + 1,
                "training_time": training_time,
                "psnr": psnr
            })
            print(f"    Training time: {training_time:.2f}s, PSNR: {psnr}")
        else:
            print(f"    Training failed for run {run + 1}")
    
    # Test loss-based method
    print("\\nTesting LOSS-BASED VIEW SELECTION...")
    for run in range(args.runs):
        print(f"  Run {run + 1}/{args.runs}")
        
        output_dir = f"./test_results/loss_judge_{dataset_name}_run{run}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_text, training_time = run_training(
            args.dataset, output_dir, use_loss_judge=True, iterations=args.iterations,
            num_selected_views=args.num_selected_views, selection_interval=args.selection_interval
        )
        
        if output_text:
            psnr = extract_psnr_from_output(output_text)
            results["loss_judge"].append({
                "run": run + 1,
                "training_time": training_time,
                "psnr": psnr
            })
            print(f"    Training time: {training_time:.2f}s, PSNR: {psnr}")
        else:
            print(f"    Training failed for run {run + 1}")
    
    # Save results
    results_file = f"./test_results/comparison_{dataset_name}_{args.iterations}iter.json"
    os.makedirs("./test_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results["baseline"]:
        baseline_times = [r["training_time"] for r in results["baseline"] if r["training_time"]]
        baseline_psnrs = [r["psnr"] for r in results["baseline"] if r["psnr"]]
        
        print(f"BASELINE (Random Sampling):")
        if baseline_times:
            print(f"  Avg Training Time: {sum(baseline_times)/len(baseline_times):.2f}s")
        if baseline_psnrs:
            print(f"  Avg PSNR: {sum(baseline_psnrs)/len(baseline_psnrs):.4f}")
    
    if results["loss_judge"]:
        loss_judge_times = [r["training_time"] for r in results["loss_judge"] if r["training_time"]]
        loss_judge_psnrs = [r["psnr"] for r in results["loss_judge"] if r["psnr"]]
        
        print(f"LOSS-BASED VIEW SELECTION:")
        if loss_judge_times:
            print(f"  Avg Training Time: {sum(loss_judge_times)/len(loss_judge_times):.2f}s")
        if loss_judge_psnrs:
            print(f"  Avg PSNR: {sum(loss_judge_psnrs)/len(loss_judge_psnrs):.4f}")
        
        # Compare with baseline
        if baseline_times and loss_judge_times:
            time_improvement = (sum(baseline_times)/len(baseline_times) - sum(loss_judge_times)/len(loss_judge_times))
            print(f"  Time Difference: {time_improvement:+.2f}s")
            
        if baseline_psnrs and loss_judge_psnrs:
            psnr_improvement = (sum(loss_judge_psnrs)/len(loss_judge_psnrs) - sum(baseline_psnrs)/len(baseline_psnrs))
            print(f"  PSNR Improvement: {psnr_improvement:+.4f}")
    
    print(f"\\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()