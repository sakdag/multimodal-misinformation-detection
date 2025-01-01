import os
import wandb
import torch
import logging
import argparse
import yaml
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import gc
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
#from sklearn.metrics import confusion_matrix

from src.model.model import MisinformationDetectionModel
from src.model.dataset import get_dataloader
from transformers import AutoTokenizer, AutoModel, Swinv2Model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train misinformation detection model')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Model settings
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--fused_attn', action='store_true', help='use fused attention')
    
    # Paths
    parser.add_argument('--train_data', type=str, default="./data/preprocessed/train.csv", help='path to training data')
    parser.add_argument('--val_data', type=str, help='path to validation data')
    parser.add_argument('--text_encoder', type=str, default='microsoft/deberta-v3-xsmall', help='text encoder model')
    parser.add_argument('--output_dir', type=str, default="./results", help='output directory')
    
    # Saving and logging
    parser.add_argument('--save_every', type=int, default=2000, help='save model every N iterations')
    parser.add_argument('--log_every', type=int, default=100, help='log metrics every N iterations')
    parser.add_argument('--wandb_project', type=str, default='misinformation-detection', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
    
    # Model freezing
    parser.add_argument('--freeze_text', action='store_true', help='freeze text encoder')
    parser.add_argument('--freeze_image', action='store_true', help='freeze image encoder')
    
    # Validation arguments
    parser.add_argument('--validate_every_epoch', action='store_true',
                       help='run validation after each epoch')
    parser.add_argument('--save_best', action='store_true',
                       help='save best model based on validation metric')
    parser.add_argument('--best_metric', type=str, default='avg_f1',
                       choices=['avg_f1', 'avg_accuracy', 'text_text_f1', 'text_image_f1', 'image_text_f1', 'image_image_f1'],
                       help='metric to track for best model saving')
    
    # Add these lines
    parser.add_argument('--log_confusion_matrix', action='store_true',
                       help='whether to log confusion matrices')
    parser.add_argument('--log_confusion_matrix_every', type=int, default=1000,
                       help='log confusion matrix every N steps')
    
    # Add new argument for pre-embedded data
    parser.add_argument('--pre_embed', action='store_true',
                       help='use pre-computed embeddings instead of raw data')
    
    # Update default input dimensions
    parser.add_argument('--text_input_dim', type=int, default=384,  # DeBERTa-v3-xsmall hidden size
                       help='input dimension for text')
    parser.add_argument('--image_input_dim', type=int, default=1024,  # Swinv2-base hidden size
                       help='input dimension for image')
    
    return parser.parse_args()

'''
def plot_confusion_matrix(cm, classes, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig'''

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_metrics(predictions, labels):
    """Compute accuracy and F1 score"""
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return accuracy, f1

def train_epoch(model, train_loader, optimizer, tokenizer, text_encoder, image_encoder, criterion, device, epoch, args, global_step):
    model.train()
    
    # Initialize metric trackers for each path
    path_predictions = {
        'text_text': [], 'text_image': [], 
        'image_text': [], 'image_image': []
    }
    path_labels = {
        'text_text': [], 'text_image': [], 
        'image_text': [], 'image_image': []
    }
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:
        optimizer.zero_grad()
        labels = batch['labels'].to(device)

        if args.pre_embed:
            # Use pre-computed embeddings
            claim_text_embeds = batch['claim_text_embeds'].to(device)
            doc_text_embeds = batch['doc_text_embeds'].to(device)
            claim_image_embeds = batch['claim_image_embeds'].to(device)
            doc_image_embeds = batch['doc_image_embeds'].to(device)
        else:
            # Original code for computing embeddings
            with torch.no_grad():
                claim_text_inputs = tokenizer(batch['claim'], truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
                claim_text_embeds = text_encoder(**claim_text_inputs).last_hidden_state

                doc_text_inputs = tokenizer(batch['document'], truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
                doc_text_embeds = text_encoder(**doc_text_inputs).last_hidden_state

                claim_image_embeds = image_encoder(batch['claim_image'].to(device)).last_hidden_state
                doc_image_embeds = image_encoder(batch['document_image'].to(device)).last_hidden_state

        # Forward pass
        (y_t_t, y_t_i), (y_i_t, y_i_i) = model(
            X_t=claim_text_embeds,
            X_i=claim_image_embeds,
            E_t=doc_text_embeds,
            E_i=doc_image_embeds
        )

        # Calculate and log losses immediately
        outputs = {
            'text_text': y_t_t,
            'text_image': y_t_i,
            'image_text': y_i_t,
            'image_image': y_i_i
        }

        losses = {}
        total_loss = 0
        
        # Calculate losses and log immediately
        for idx, (path, output) in enumerate(outputs.items()):
            if output is not None:
                path_loss = criterion(output, labels[:, idx])
                losses[f'train/{path}_loss'] = path_loss.item()
                total_loss += path_loss
                
                # Store predictions and labels for metric calculation
                pred = output.argmax(dim=-1)
                path_predictions[path].extend(pred.cpu().tolist())
                path_labels[path].extend(labels[:, idx].cpu().tolist())

        # Log losses and learning rate immediately
        wandb.log({
            'train/total_loss': total_loss.item(),
            **losses,
            'train/learning_rate': args.lr,
            'train/step': global_step
        })

        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()

        # Calculate and log metrics every log_every steps
        if global_step % args.log_every == 0:
            metrics_log = {}
            
            for path in path_predictions.keys():
                if path_predictions[path]:  # If we have predictions for this path
                    # Accuracy
                    accuracy = accuracy_score(path_labels[path], path_predictions[path])
                    metrics_log[f'train/{path}_accuracy'] = accuracy
                    
                    # F1 Score (weighted average)
                    f1 = f1_score(path_labels[path], path_predictions[path], average='weighted')
                    metrics_log[f'train/{path}_f1'] = f1
                    
                    # Per-class F1 scores
                    per_class_f1 = f1_score(path_labels[path], path_predictions[path], average=None)
                    for idx, class_f1 in enumerate(per_class_f1):
                        metrics_log[f'train/{path}_class{idx}_f1'] = class_f1

                    '''
                    # Optional: confusion matrix
                    if args.log_confusion_matrix and global_step % args.log_confusion_matrix_every == 0:
                        cm = confusion_matrix(path_labels[path], path_predictions[path])
                        fig = plot_confusion_matrix(cm, ['Support', 'NEI', 'Refute'], path)
                        metrics_log[f'train/{path}_confusion_matrix'] = wandb.Image(fig)
                        plt.close(fig)
                    '''

            # Log accumulated metrics
            wandb.log(metrics_log)

            # Reset predictions and labels after logging
            path_predictions = {k: [] for k in path_predictions}
            path_labels = {k: [] for k in path_labels}

            # Update progress bar with latest metrics
            avg_acc = np.mean([metrics_log[k] for k in metrics_log if 'accuracy' in k])
            avg_f1 = np.mean([metrics_log[k] for k in metrics_log if k.endswith('_f1')])
            pbar.set_description(
                f"Epoch {epoch} | Loss: {total_loss.item():.4f} | "
                f"Acc: {avg_acc:.4f} | F1: {avg_f1:.4f}"
            )

        # Save model based on iterations
        if global_step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint-{epoch}-{global_step}')
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'global_step': global_step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_path, 'model.pt'))

        global_step += 1

    return total_loss.item(), global_step

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    path_losses = {'text_text': 0, 'text_image': 0, 'image_text': 0, 'image_image': 0}
    
    predictions = {
        'text_text': [], 'text_image': [], 
        'image_text': [], 'image_image': []
    }
    labels_list = {
        'text_text': [], 'text_image': [], 
        'image_text': [], 'image_image': []
    }

    for batch in tqdm(val_loader, desc="Evaluating"):
        # Process batch similar to training
        claim_text = batch['claim']
        claim_image = batch['claim_image'].to(device)
        document_text = batch['document']
        document_image = batch['document_image'].to(device)
        labels = batch['labels'].to(device)

        claim_text_inputs = model.tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt").to(device)
        claim_text_embeds = model.text_encoder(**claim_text_inputs).last_hidden_state

        doc_text_inputs = model.tokenizer(document_text, truncation=True, padding=True, return_tensors="pt").to(device)
        doc_text_embeds = model.text_encoder(**doc_text_inputs).last_hidden_state

        (y_t_t, y_t_i), (y_i_t, y_i_i) = model(
            X_t=claim_text_embeds,
            X_i=claim_image,
            E_t=doc_text_embeds,
            E_i=document_image
        )

        # Calculate losses
        outputs = {
            'text_text': y_t_t,
            'text_image': y_t_i,
            'image_text': y_i_t,
            'image_image': y_i_i
        }

        for idx, (path, output) in enumerate(outputs.items()):
            loss = criterion(output, labels[:, idx])
            path_losses[path] += loss.item()
            
            pred = output.argmax(dim=-1)
            predictions[path].extend(pred.cpu().tolist())
            labels_list[path].extend(labels[:, idx].cpu().tolist())

    # Calculate metrics
    metrics = {}
    for path in predictions.keys():
        accuracy, f1 = compute_metrics(predictions[path], labels_list[path])
        metrics[f'{path}_accuracy'] = accuracy
        metrics[f'{path}_f1'] = f1

    avg_loss = {k: v / len(val_loader) for k, v in path_losses.items()}
    
    return avg_loss, metrics

def main(args):
    # Initialize wandb
    wbkwargs = {'project': args.wandb_project, 'config': vars(args),
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': 'online', 'save_code': True}
    wandb.init(**wbkwargs)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and seed
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Only initialize encoders if not using pre-computed embeddings
    tokenizer = None
    text_encoder = None
    image_encoder = None
    
    if not args.pre_embed:
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
        text_encoder = AutoModel.from_pretrained(args.text_encoder).to(device)
        image_encoder = Swinv2Model.from_pretrained('microsoft/swinv2-base-patch4-window8-256').to(device)
        
        # Freeze encoders and set to eval mode
        text_encoder.eval()
        image_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
        for param in image_encoder.parameters():
            param.requires_grad = False
    
    # Initialize model with separate text and image dimensions
    model = MisinformationDetectionModel(
        text_input_dim=args.text_input_dim,
        image_input_dim=args.image_input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        mlp_ratio=args.mlp_ratio,
        fused_attn=args.fused_attn
    ).to(device)

    # Only create optimizer for the main model
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Validate arguments
    if args.validate_every_epoch and not args.val_data:
        raise ValueError("--val_data must be specified when --validate_every_epoch is set")

    # Get train dataloader with appropriate dataset class
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pre_embed=args.pre_embed
    )

    # Get validation dataloader only if needed
    val_loader = None
    if args.validate_every_epoch:
        logger.info("Creating validation dataloader...")
        val_loader = get_dataloader(
            args.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    best_metric = float('-inf')
    
    for epoch in range(args.epochs):
        # Training
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer,
            tokenizer, text_encoder, image_encoder, criterion,
            device, epoch, args, global_step
        )
        
        # Validation
        if args.validate_every_epoch:
            val_losses, val_metrics = evaluate(model, val_loader, criterion, device)
            
            # Log validation metrics
            wandb.log({
                'val/loss': sum(val_losses.values()) / len(val_losses),
                **{f'val/{k}_loss': v for k, v in val_losses.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'epoch': epoch,
                'global_step': global_step
            })
            
            # Save best model if requested
            if args.save_best:
                current_metric = None
                if args.best_metric == 'avg_f1':
                    current_metric = np.mean([v for k, v in val_metrics.items() if 'f1' in k])
                elif args.best_metric == 'avg_accuracy':
                    current_metric = np.mean([v for k, v in val_metrics.items() if 'accuracy' in k])
                else:
                    current_metric = val_metrics.get(args.best_metric)
                
                if current_metric is not None and current_metric > best_metric:
                    best_metric = current_metric
                    logger.info(f"New best model with {args.best_metric}: {best_metric:.4f}")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        f'{args.best_metric}': best_metric,
                    }, os.path.join(args.output_dir, 'best_model.pt'))
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    
    # Print arguments
    logger.info("Training arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    main(args)