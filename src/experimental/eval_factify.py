import os
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from src.model.model import MisinformationDetectionModel
from src.model.dataset import (
    get_dataloader,
    category_to_idx,
    idx_to_category,
    labels_to_category
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate factify model')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='path to test data')
    parser.add_argument('--batch_size', type=int, default=192, help='batch size for evaluation')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--output_file', type=str, default='factify_evaluation_results.csv', 
                       help='path to output CSV file')
    parser.add_argument('--plot_confusion', action='store_true', 
                       help='plot confusion matrix')
    return parser.parse_args()

def convert_labels_to_category_idx(labels):
    """Convert multi-path labels to single category index."""
    batch_categories = []
    for label_vec in labels:
        label_tuple = tuple(label_vec.cpu().tolist())
        category = labels_to_category.get(label_tuple, 'Insufficient_Text')
        category_idx = category_to_idx[category]
        batch_categories.append(category_idx)
    return torch.tensor(batch_categories, device=labels.device)

def plot_confusion_matrix(cm, categories, output_path):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calculate_weighted_accuracy(y_true, y_pred, refute_weight=4):
    """
    Calculate accuracy with higher weight for refuting samples.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        refute_weight: Weight for refuting samples (category_to_idx['Refute'] = 4)
    
    Returns:
        weighted_accuracy: Accuracy score with weighted refuting samples
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create weight array (1 for non-refute, refute_weight for refute)
    weights = np.ones_like(y_true, dtype=float)
    weights[y_true == category_to_idx['Refute']] = refute_weight
    
    # Calculate weighted accuracy
    correct = (y_true == y_pred) * weights
    weighted_accuracy = correct.sum() / weights.sum()
    
    return weighted_accuracy

def calculate_category_accuracies(y_true, y_pred, categories):
    """
    Calculate accuracy for each category separately.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        categories: List of category names
    
    Returns:
        dict: Dictionary containing accuracy for each category
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracies = {}
    for category, idx in category_to_idx.items():
        # Get samples belonging to this category
        category_mask = (y_true == idx)
        if category_mask.sum() > 0:  # Avoid division by zero
            category_correct = (y_pred[category_mask] == idx).sum()
            category_total = category_mask.sum()
            accuracies[category] = category_correct / category_total
        else:
            accuracies[category] = 0.0
            
    return accuracies

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Get labels and convert to category indices
        labels = batch['labels'].to(device)
        ground_truth = convert_labels_to_category_idx(labels)
        
        # Forward pass
        outputs = model(
            X_t=batch['claim_text_embeds'].to(device),
            X_i=batch['claim_image_embeds'].to(device),
            E_t=batch['doc_text_embeds'].to(device),
            E_i=batch['doc_image_embeds'].to(device)
        )
        
        unified_pred, _ = outputs
        predictions = unified_pred.argmax(dim=-1)
        
        all_predictions.extend(predictions.cpu().tolist())
        all_ground_truths.extend(ground_truth.cpu().tolist())
    
    return all_predictions, all_ground_truths

def save_metrics(metrics_dict, output_file):
    """Save metrics to CSV file"""
    results = pd.DataFrame([metrics_dict])
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if os.path.exists(output_file):
        existing_results = pd.read_csv(output_file)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        updated_results = results
    
    updated_results.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', {})
    
    model = MisinformationDetectionModel(
        text_input_dim=model_config.get('text_input_dim', 768),
        image_input_dim=model_config.get('image_input_dim', 1024),
        embed_dim=model_config.get('embed_dim', 256),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.1),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_classes=5,  # Fixed for factify evaluation
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        fused_attn=model_config.get('fused_attn', False),
        factify=True  # Force factify mode
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test dataloader
    test_loader = get_dataloader(
        args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pre_embed=True
    )
    
    # Evaluate model
    predictions, ground_truths = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    categories = list(category_to_idx.keys())
    
    # Calculate standard and weighted overall accuracy
    standard_accuracy = accuracy_score(ground_truths, predictions)
    weighted_accuracy = calculate_weighted_accuracy(ground_truths, predictions, refute_weight=4)
    
    # Calculate per-category accuracies
    category_accuracies = calculate_category_accuracies(ground_truths, predictions, categories)
    
    # Create metrics dictionary
    metrics = {
        'model_path': args.model_path,
        'standard_accuracy': standard_accuracy,
        'weighted_accuracy': weighted_accuracy
    }
    
    # Add per-category accuracies
    for category in categories:
        metrics[f'{category}_accuracy'] = category_accuracies[category]
    
    # Print metrics
    logger.info("\nEvaluation Results:")
    logger.info(f"Standard Accuracy: {standard_accuracy:.4f}")
    logger.info(f"Weighted Accuracy (Refute weight=4): {weighted_accuracy:.4f}")
    logger.info("\nPer-category accuracies:")
    for category in categories:
        accuracy = category_accuracies[category]
        logger.info(f"{category}: {accuracy:.4f}")
    
    # Save metrics to CSV
    save_metrics(metrics, args.output_file)
    
    # Plot confusion matrix if requested
    if args.plot_confusion:
        cm = confusion_matrix(ground_truths, predictions)
        output_dir = os.path.dirname(args.output_file)
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, categories, plot_path)
        logger.info(f"Confusion matrix saved to {plot_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)