import h5py
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Swinv2Model

logger = logging.getLogger(__name__)

@torch.no_grad()
def create_embeddings_h5(input_h5_path, output_h5_path, batch_size=32, device='cuda'):
    """
    Create a new H5 file with pre-computed embeddings from text and images.
    
    Args:
        input_h5_path (str): Path to input H5 file with raw data
        output_h5_path (str): Path where to save the new H5 file with embeddings
        batch_size (int): Batch size for processing
        device (str): Device to use for computation
    """
    logger.info(f"Creating embeddings H5 file from {input_h5_path}")
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')
    text_encoder = AutoModel.from_pretrained('microsoft/deberta-v3-xsmall').to(device)
    image_encoder = Swinv2Model.from_pretrained('microsoft/swinv2-base-patch4-window8-256').to(device)
    
    # Set models to eval mode
    text_encoder.eval()
    image_encoder.eval()
    
    # Open input H5 file
    with h5py.File(input_h5_path, 'r') as in_f, h5py.File(output_h5_path, 'w') as out_f:
        total_samples = len(in_f.keys())
        
        # Process in batches
        for batch_start in tqdm(range(0, total_samples, batch_size)):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_indices = range(batch_start, batch_end)
            
            # Collect batch data
            claim_texts = []
            doc_texts = []
            claim_images = []
            doc_images = []
            labels = []
            
            for idx in batch_indices:
                sample = in_f[str(idx)]
                claim_texts.append(sample['claim'][()].decode())
                doc_texts.append(sample['document'][()].decode())
                claim_images.append(torch.from_numpy(sample['claim_image'][()]))
                doc_images.append(torch.from_numpy(sample['document_image'][()]))
                labels.append(sample['labels'][()])
            
            # Convert to tensors
            claim_images = torch.stack(claim_images).to(device)
            doc_images = torch.stack(doc_images).to(device)
            
            # Get text embeddings with fixed sequence length
            claim_text_inputs = tokenizer(
                claim_texts, 
                truncation=True, 
                padding='max_length',  # Changed to max_length
                return_tensors="pt", 
                max_length=512
            ).to(device)
            
            doc_text_inputs = tokenizer(
                doc_texts, 
                truncation=True, 
                padding='max_length',  # Changed to max_length
                return_tensors="pt", 
                max_length=512
            ).to(device)
            
            claim_text_embeds = text_encoder(**claim_text_inputs).last_hidden_state
            doc_text_embeds = text_encoder(**doc_text_inputs).last_hidden_state
            
            # Verify shapes
            assert claim_text_embeds.shape[1] == 512, f"Unexpected claim text shape: {claim_text_embeds.shape}"
            assert doc_text_embeds.shape[1] == 512, f"Unexpected doc text shape: {doc_text_embeds.shape}"
            
            # Get image embeddings
            claim_image_embeds = image_encoder(claim_images).last_hidden_state
            doc_image_embeds = image_encoder(doc_images).last_hidden_state
            
            # Store embeddings and labels
            for batch_idx, idx in enumerate(batch_indices):
                sample_group = out_f.create_group(str(idx))
                
                # Store embeddings
                sample_group.create_dataset('claim_text_embeds', 
                                         data=claim_text_embeds[batch_idx].cpu().numpy())
                sample_group.create_dataset('doc_text_embeds', 
                                         data=doc_text_embeds[batch_idx].cpu().numpy())
                sample_group.create_dataset('claim_image_embeds', 
                                         data=claim_image_embeds[batch_idx].cpu().numpy())
                sample_group.create_dataset('doc_image_embeds', 
                                         data=doc_image_embeds[batch_idx].cpu().numpy())
                
                # Store labels
                sample_group.create_dataset('labels', data=labels[batch_idx])
    
    logger.info(f"Created embeddings H5 file at {output_h5_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    create_embeddings_h5(
        input_h5_path='data/preprocessed/train.h5',
        output_h5_path='data/preprocessed/train_embeddings.h5',
        batch_size=32,
        device='cuda:0'
    )