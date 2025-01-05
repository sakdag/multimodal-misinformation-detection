import os
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from src.model.dataset import get_dataloader, simplified_idx_to_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BLOOMZ on misinformation detection')
    parser.add_argument('--model_name', type=str, default='bigscience/bloomz-560m',
                       help='BLOOMZ model to use')
    parser.add_argument('--test_data', type=str, required=True,
                       help='path to test data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='batch size for evaluation')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='number of workers for data loading')
    parser.add_argument('--output_file', type=str, default='bloom_evaluation_results.csv',
                       help='path to output CSV file')
    return parser.parse_args()

def get_label_from_response(response):
    """Extract the label from model response and normalize it."""
    # Extract last line and clean up
    label = response.strip().split('\n')[-1].lower()
    
    # Normalize the label
    if 'support' in label:
        return 0  # Support
    elif 'refute' in label or 'false' in label:
        return 2  # Refute
    else:
        return 1  # NEI

def create_prompt(claim, evidence):
    """Create prompt for the model combining claim and evidence."""
    return f"""You are an expert fact-checker. Your task is to verify if the given evidence supports or refutes the claim, or if there is not enough information to make a determination.

    Claim: {claim}

    Evidence: {evidence}

    Based on the evidence, the claim is (respond with exactly one of: SUPPORT / NOT ENOUGH INFORMATION / REFUTE):"""

@torch.no_grad()
def evaluate_model(model, tokenizer, test_loader, device):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Convert multi-path labels to simplified 3-class ground truth
        ground_truths = batch['labels'][:, 0].tolist()  
        
        for i in range(len(ground_truths)):
            # Get claim and evidence text
            claim = batch['claim'][i]
            evidence = batch['document'][i]
            
            # Create prompt
            prompt = create_prompt(claim, evidence)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                             max_length=400).to(device)  # Reduced from 512 to 400
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Allow up to 64 new tokens for response
                temperature=0.1,
                do_sample=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get prediction
            prediction = get_label_from_response(response)
            
            all_predictions.append(prediction)
            all_ground_truths.append(ground_truths[i])
    
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
    
    # Load model and tokenizer
    logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Get test dataloader
    test_loader = get_dataloader(
        args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pre_embed=False  # We need raw text for prompting
    )
    
    # Evaluate model
    predictions, ground_truths = evaluate_model(model, tokenizer, test_loader, device)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths,
        predictions,
        labels=[0, 1, 2],  # Support, NEI, Refute
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
        'model_name': args.model_name,
        'micro_f1': micro_f1
    }
    
    # Add class-specific metrics
    class_names = ['Support', 'NEI', 'Refute']
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