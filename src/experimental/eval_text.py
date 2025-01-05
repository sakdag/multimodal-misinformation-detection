import os
import torch
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from datetime import datetime

from src.model.model import MisinformationDetectionModel
from src.model.dataset import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text-only misinformation detection')
    parser.add_argument('--model_path', type=str, required=True,
                       help='path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='path to test data')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='batch size for evaluation')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='number of workers for data loading')
    parser.add_argument('--output_file', type=str, default='text_only_results.csv',
                       help='path to output CSV file')
    return parser.parse_args()

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

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Get ground truth (using first label as we're in text-only mode)
        ground_truths = batch['labels'][:, 0].tolist()
        
        # Get pre-computed text embeddings
        claim_embeds = batch['claim_text_embeds'].to(device)
        evidence_embeds = batch['doc_text_embeds'].to(device)
        
        # Forward pass through model
        predictions, _ = model(X_t=claim_embeds, E_t=evidence_embeds)
        
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1).cpu().tolist()
        
        all_predictions.extend(pred_classes)
        all_ground_truths.extend(ground_truths)
    
    return all_predictions, all_ground_truths

def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', {})
    
    # Initialize model in text-only mode
    model = MisinformationDetectionModel(
        text_input_dim=768,  # DeBERTa hidden size
        embed_dim=model_config.get('embed_dim', 256),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.1),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_classes=3,  # Support, NEI, Refute
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        text_only=True  # Enable text-only mode
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test dataloader with pre-computed embeddings
    test_loader = get_dataloader(
        args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pre_embed=True  # Use pre-computed embeddings
    )
    
    # Evaluate model
    predictions, ground_truths = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths,
        predictions,
        labels=[0, 1, 2],  # Support, NEI, Refute
        average=None
    )
    
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Calculate micro-F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        ground_truths,
        predictions,
        average='micro'
    )
    
    # Create metrics dictionary
    metrics = {
        'model_path': args.model_path,
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'support_precision': precision[0],
        'support_recall': recall[0],
        'support_f1': f1[0],
        'nei_precision': precision[1],
        'nei_recall': recall[1],
        'nei_f1': f1[1],
        'refute_precision': precision[2],
        'refute_recall': recall[2],
        'refute_f1': f1[2]
    }
    
    # Print metrics
    logger.info("\nEvaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Micro-F1: {micro_f1:.4f}")
    
    class_names = ['Support', 'NEI', 'Refute']
    for i, name in enumerate(class_names):
        logger.info(f"{name:<20} - P: {precision[i]:.4f}, "
                   f"R: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    # Save metrics to CSV
    save_metrics(metrics, args.output_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)