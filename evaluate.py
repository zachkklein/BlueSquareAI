#!/usr/bin/env python3
"""
Comprehensive evaluation script for antisemitism detection system.

This script evaluates the accuracy of the classification system using
various metrics including MAE, correlation, precision, recall, and F1.
"""

import json
import numpy as np
from typing import List, Dict
from pipeline.aggregate import classify_text, classify_text_async
import asyncio
from eval_data import evaluation_data

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib or sklearn not available. Plots will be skipped.")


def calculate_metrics(predictions: List[float], labels: List[float]) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for classification system.

    Parameters
    ----------
    predictions
        List of predicted risk scores (0.0-1.0).
    labels
        List of ground truth labels (0.0-1.0).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - r2: R² score
        - correlation: Pearson correlation coefficient
        - precision: Precision at threshold 0.5
        - recall: Recall at threshold 0.5
        - f1: F1 score at threshold 0.5
        - accuracy: Accuracy at threshold 0.5
        - tp, fp, fn, tn: Confusion matrix components
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Regression metrics
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    correlation = np.corrcoef(labels, predictions)[0, 1]
    
    # Classification metrics (using threshold 0.5)
    binary_pred = (predictions >= 0.5).astype(int)
    binary_label = (labels >= 0.5).astype(int)
    
    tp = np.sum((binary_pred == 1) & (binary_label == 1))
    fp = np.sum((binary_pred == 1) & (binary_label == 0))
    fn = np.sum((binary_pred == 0) & (binary_label == 1))
    tn = np.sum((binary_pred == 0) & (binary_label == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "correlation": correlation,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


async def evaluate_system_async(eval_data: List[Dict]) -> Dict:
    """Evaluate the system on the evaluation dataset using async processing.

    Parameters
    ----------
    eval_data
        List of evaluation examples, each containing "text" and "label" keys.

    Returns
    -------
    Dict
        Evaluation results containing:
        - metrics: Dictionary of calculated metrics
        - results: List of individual prediction results
        - predictions: List of predicted scores
        - labels: List of ground truth labels
    """
    predictions = []
    labels = []
    results = []
    
    print("Evaluating system on {} examples (using async for speed)...\n".format(len(eval_data)))
    
    # Process all examples in parallel batches
    batch_size = 5  # Process 5 at a time to avoid overwhelming the API
    tasks = []
    
    for i, example in enumerate(eval_data):
        text = example["text"]
        label = example["label"]
        tasks.append((i, text, label, classify_text_async(text)))
    
    # Process in batches
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        batch_results = await asyncio.gather(*[task[3] for task in batch], return_exceptions=True)
        
        for (i, text, label, _), result in zip(batch, batch_results):
            print(f"[{i+1}/{len(eval_data)}] Processing: {text[:60]}...")
            
            if isinstance(result, Exception):
                print(f"  ERROR: {str(result)}")
                predictions.append(0.0)
                labels.append(label)
                results.append({
                    "text": text,
                    "true_label": label,
                    "predicted_score": 0.0,
                    "error": str(result)
                })
            else:
                pred_score = result["risk_score"]
                predictions.append(pred_score)
                labels.append(label)
                
                results.append({
                    "text": text,
                    "true_label": label,
                    "predicted_score": pred_score,
                    "verdict": result["verdict"],
                    "trope": result["trope"],
                    "trope_strength": result.get("trope_strength", 0.0),
                    "error": abs(label - pred_score)
                })
                
                print(f"  True: {label:.2f}, Predicted: {pred_score:.2f}, Error: {abs(label - pred_score):.2f}")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    return {
        "metrics": metrics,
        "results": results,
        "predictions": predictions,
        "labels": labels
    }


def evaluate_system(eval_data: List[Dict]) -> Dict:
    """Evaluate the system on the evaluation dataset (synchronous wrapper).

    Parameters
    ----------
    eval_data
        List of evaluation examples, each containing "text" and "label" keys.

    Returns
    -------
    Dict
        Evaluation results containing:
        - metrics: Dictionary of calculated metrics
        - results: List of individual prediction results
        - predictions: List of predicted scores
        - labels: List of ground truth labels
    """
    return asyncio.run(evaluate_system_async(eval_data))
    
    return {
        "metrics": metrics,
        "results": results,
        "predictions": predictions,
        "labels": labels
    }


def plot_results(eval_results: Dict, save_path: str = "evaluation_results.png"):
    """Create visualization plots for evaluation results."""
    if not HAS_PLOTTING:
        print("Skipping plots - matplotlib/sklearn not available")
        return None
        
    predictions = eval_results["predictions"]
    labels = eval_results["labels"]
    metrics = eval_results["metrics"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Antisemitism Detection System Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Predicted vs True
    ax1 = axes[0, 0]
    ax1.scatter(labels, predictions, alpha=0.6, s=100)
    ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True Label', fontsize=12)
    ax1.set_ylabel('Predicted Score', fontsize=12)
    ax1.set_title(f'Predicted vs True Scores\nCorrelation: {metrics["correlation"]:.3f}, R²: {metrics["r2"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    errors = [abs(p - l) for p, l in zip(predictions, labels)]
    ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(metrics["mae"], color='r', linestyle='--', lw=2, label=f'MAE: {metrics["mae"]:.3f}')
    ax2.set_xlabel('Absolute Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = axes[1, 0]
    residuals = [p - l for p, l in zip(predictions, labels)]
    ax3.scatter(labels, residuals, alpha=0.6, s=100)
    ax3.axhline(0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('True Label', fontsize=12)
    ax3.set_ylabel('Residual (Predicted - True)', fontsize=12)
    ax3.set_title('Residual Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Regression Metrics:
    • MAE: {metrics['mae']:.3f}
    • RMSE: {metrics['rmse']:.3f}
    • R²: {metrics['r2']:.3f}
    • Correlation: {metrics['correlation']:.3f}
    
    Classification Metrics (threshold=0.5):
    • Accuracy: {metrics['accuracy']:.3f}
    • Precision: {metrics['precision']:.3f}
    • Recall: {metrics['recall']:.3f}
    • F1 Score: {metrics['f1']:.3f}
    
    Confusion Matrix:
    • TP: {metrics['tp']}, FP: {metrics['fp']}
    • FN: {metrics['fn']}, TN: {metrics['tn']}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    return fig


def analyze_by_trope(results: List[Dict]) -> Dict:
    """Analyze performance by detected trope."""
    trope_stats = {}
    
    for result in results:
        trope = result.get("trope", "none")
        if trope not in trope_stats:
            trope_stats[trope] = {
                "count": 0,
                "errors": [],
                "true_scores": [],
                "pred_scores": []
            }
        
        trope_stats[trope]["count"] += 1
        if "error" in result and isinstance(result["error"], (int, float)):
            trope_stats[trope]["errors"].append(result["error"])
            trope_stats[trope]["true_scores"].append(result["true_label"])
            trope_stats[trope]["pred_scores"].append(result["predicted_score"])
    
    # Calculate metrics per trope
    for trope, stats in trope_stats.items():
        if stats["errors"]:
            stats["mean_error"] = np.mean(stats["errors"])
            stats["mean_true"] = np.mean(stats["true_scores"])
            stats["mean_pred"] = np.mean(stats["pred_scores"])
        else:
            stats["mean_error"] = 0.0
            stats["mean_true"] = 0.0
            stats["mean_pred"] = 0.0
    
    return trope_stats


def print_detailed_results(eval_results: Dict):
    """Print detailed evaluation results."""
    metrics = eval_results["metrics"]
    results = eval_results["results"]
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\n--- REGRESSION METRICS ---")
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"R² Score:                      {metrics['r2']:.4f}")
    print(f"Correlation:                   {metrics['correlation']:.4f}")
    
    print("\n--- CLASSIFICATION METRICS (threshold=0.5) ---")
    print(f"Accuracy:                      {metrics['accuracy']:.4f}")
    print(f"Precision:                     {metrics['precision']:.4f}")
    print(f"Recall:                        {metrics['recall']:.4f}")
    print(f"F1 Score:                      {metrics['f1']:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Positives (TP):  {metrics['tp']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    print(f"True Negatives (TN):  {metrics['tn']}")
    
    # Per-trope analysis
    trope_stats = analyze_by_trope(results)
    print("\n--- PERFORMANCE BY TROPE ---")
    for trope, stats in sorted(trope_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        if stats["count"] > 0:
            print(f"\n{trope}:")
            print(f"  Count: {stats['count']}")
            if stats["errors"]:
                print(f"  Mean Error: {stats['mean_error']:.3f}")
                print(f"  Mean True Score: {stats['mean_true']:.3f}")
                print(f"  Mean Predicted Score: {stats['mean_pred']:.3f}")
    
    print("\n--- WORST PREDICTIONS (by absolute error) ---")
    sorted_results = sorted(results, key=lambda x: x.get("error", 0) if isinstance(x.get("error"), (int, float)) else 0, reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        error = result.get("error", 0) if isinstance(result.get("error"), (int, float)) else 0
        print(f"\n{i+1}. Error: {error:.3f}")
        print(f"   Text: {result['text'][:80]}...")
        print(f"   True: {result['true_label']:.2f}, Predicted: {result['predicted_score']:.2f}")
        print(f"   Trope: {result.get('trope', 'N/A')}, Verdict: {result.get('verdict', 'N/A')}")
    
    print("\n" + "="*80)


def main():
    """
    Main evaluation function.
    
    Runs comprehensive evaluation and generates reports.
    """
    print("Starting evaluation...")
    print("="*80)
    
    # Run evaluation
    eval_results = evaluate_system(evaluation_data)
    
    # Print results
    print_detailed_results(eval_results)
    
    # Create visualizations
    if HAS_PLOTTING:
        try:
            plot_results(eval_results)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    else:
        print("\nSkipping visualization (matplotlib/sklearn not available)")
    
    # Save detailed results to JSON
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
    
    return eval_results


if __name__ == "__main__":
    main()

