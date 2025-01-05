import os
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime

from src.model.model import MisinformationDetectionModel
from src.model.dataset import (
    get_dataloader,
    category_to_idx,
    idx_to_category,
    labels_to_category,
    simplified_category_to_idx,
    simplified_idx_to_category,
    convert_to_simplified_category
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate misinformation detection model')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='path to test data')
    parser.add_argument('--batch_size', type=int, default=192, help='batch size for evaluation')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv', 
                       help='path to output CSV file')
    parser.add_argument('--factify', action='store_true',
                       help='use single MLP head instead of separate paths')
    parser.add_argument('--simplified_classes', action='store_true',
                       help='use 3-class classification (Support, NEI, Refute)')
    return parser.parse_args()

def convert_labels_to_category_idx(labels, simplified=False):
    """Convert multi-path labels to single category index."""
    batch_categories = []
    for label_vec in labels:
        label_tuple = tuple(label_vec.cpu().tolist())
        category = labels_to_category.get(label_tuple, 'Insufficient_Text')
        category_idx = category_to_idx[category]
        
        if simplified:
            # Convert 5-class to 3-class
            category_idx = convert_to_simplified_category(category_idx)
            
        batch_categories.append(category_idx)
    return torch.tensor(batch_categories, device=labels.device)

@torch.no_grad()
def evaluate_model(model, test_loader, device, args):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        labels = batch['labels'].to(device)
        
        if args.factify:
            # Convert multi-path labels to category indices
            ground_truth = convert_labels_to_category_idx(labels, simplified=args.simplified_classes)
        
        # Get predictions using pre-embedded inputs only
        outputs = model(
            X_t=batch['claim_text_embeds'].to(device),
            X_i=batch['claim_image_embeds'].to(device),
            E_t=batch['doc_text_embeds'].to(device),
            E_i=batch['doc_image_embeds'].to(device)
        )
        
        if args.factify:
            unified_pred, _ = outputs
            predictions = unified_pred.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_ground_truths.extend(ground_truth.cpu().tolist())
        else:
            (y_t_t, y_t_i), (y_i_t, y_i_i) = outputs
            
            # Get predictions for text-text and image-image paths
            text_text_pred = y_t_t.argmax(dim=-1)
            text_image_pred = y_t_i.argmax(dim=-1)
            image_text_pred = y_i_t.argmax(dim=-1)
            image_image_pred = y_i_i.argmax(dim=-1)
            
            # Convert to unified predictions
            batch_predictions = [
                get_unified_prediction(tt.item(), ti.item(), it.item(), ii.item())
                for tt, ti, it, ii in zip(text_text_pred, text_image_pred, image_text_pred, image_image_pred)
            ]
            
            # Get ground truth labels
            batch_ground_truths = [
                get_ground_truth_category(idx_to_category[label])
                for label in labels[:, 0].cpu().numpy()  # Use first label as category indicator
            ]
            
            all_predictions.extend(batch_predictions)
            all_ground_truths.extend(batch_ground_truths)
    
    return all_predictions, all_ground_truths

def save_metrics(metrics_dict, output_file):
    """Save metrics to CSV file"""
    # Create results DataFrame
    results = pd.DataFrame([metrics_dict])
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Append or create new file
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
    
    # Set number of classes based on simplified_classes flag
    num_classes = 3 if args.simplified_classes else 5
    
    model = MisinformationDetectionModel(
        text_input_dim=model_config.get('text_input_dim', 768),
        image_input_dim=model_config.get('image_input_dim', 1024),
        embed_dim=model_config.get('embed_dim', 256),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.1),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_classes=num_classes,
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        fused_attn=model_config.get('fused_attn', False),
        factify=args.factify
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
    predictions, ground_truths = evaluate_model(model, test_loader, device, args)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, 
        predictions, 
        labels=list(range(num_classes)),  # Use appropriate number of classes
        average=None
    )
    
    # Calculate micro-F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        ground_truths,
        predictions,
        average='micro'
    )
    
    # Create metrics dictionary
    metrics = {
        'model_path': args.model_path,
        'micro_f1': micro_f1
    }
    
    # Add class-specific metrics
    if args.simplified_classes:
        class_names = ['Support', 'NEI', 'Refute']
    else:
        class_names = ['Support_Text', 'Support_Multimodal', 'Insufficient_Text', 'Insufficient_Multimodal', 'Refute']
    
    for i, name in enumerate(class_names):
        metrics[f'{name}_precision'] = precision[i]
        metrics[f'{name}_recall'] = recall[i]
        metrics[f'{name}_f1'] = f1[i]
    
    # Print metrics
    logger.info("\nEvaluation Results:")
    for name in class_names:
        logger.info(f"{name:<20} - P: {metrics[f'{name}_precision']:.4f}, "
                   f"R: {metrics[f'{name}_recall']:.4f}, F1: {metrics[f'{name}_f1']:.4f}")
    logger.info(f"Micro-F1: {micro_f1:.4f}")
    
    # Save metrics to CSV
    save_metrics(metrics, args.output_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)