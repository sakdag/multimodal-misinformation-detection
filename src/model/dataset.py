import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
])

# Updated category mapping for multi-label classification
# Each category maps to (text-text, text-image, image-text, image-image) labels
# 0: Support, 1: NEI (Not Enough Information), 2: Refute
category_to_labels = {
    'Support_Text': [0, 1, 1, 1],        # Support only for text-text
    'Support_Multimodal': [0, 0, 0, 0],  # Support for all paths
    'Insufficient_Text': [1, 1, 1, 1],   # NEI for all paths
    'Insufficient_Multimodal': [1, 1, 1, 0],  # Support for cross-modal paths, NEI for others
    'Refute': [2, 2, 2, 2]              # Refute for all paths
}

# Add reverse mapping from label patterns to categories
labels_to_category = {
    tuple([0, 1, 1, 1]): 'Support_Text',
    tuple([0, 0, 0, 0]): 'Support_Multimodal',
    tuple([1, 1, 1, 1]): 'Insufficient_Text',
    tuple([1, 1, 1, 0]): 'Insufficient_Multimodal',
    tuple([2, 2, 2, 2]): 'Refute'
}

# Add category to index mapping for classification
category_to_idx = {
    'Support_Text': 0,
    'Support_Multimodal': 1,
    'Insufficient_Text': 2,
    'Insufficient_Multimodal': 3,
    'Refute': 4
}

# Add index to category mapping for converting predictions back to categories
idx_to_category = {v: k for k, v in category_to_idx.items()}

# Add simplified category mappings
simplified_category_mapping = {
    'Support_Text': 'Support',
    'Support_Multimodal': 'Support',
    'Insufficient_Text': 'NEI',
    'Insufficient_Multimodal': 'NEI',
    'Refute': 'Refute'
}

simplified_category_to_idx = {
    'Support': 0,
    'NEI': 1,
    'Refute': 2
}

simplified_idx_to_category = {v: k for k, v in simplified_category_to_idx.items()}

def convert_to_simplified_category(category_idx):
    """Convert 5-class category index to 3-class simplified index."""
    category = idx_to_category[category_idx]
    simplified_category = simplified_category_mapping[category]
    return simplified_category_to_idx[simplified_category]

def prepare_h5_dataset(csv_path, h5_path, enriched=False):
    """
    Prepare h5 dataset from CSV file where each index contains complete sample data.
    Skips samples where either claim image or evidence image is missing.
    
    Args:
        csv_path: Path to input CSV file
        h5_path: Path to output H5 file
        enriched: If True, use enriched claim/evidence text fields
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    # Read CSV file
    columns = ['claim_enriched' if enriched else 'claim',
              'claim_image',
              'evidence_enriched' if enriched else 'evidence', 
              'evidence_image',
              'category']
    df = pd.read_csv(csv_path, index_col=0)[columns]
    
    with h5py.File(h5_path, 'w') as f:
        # Process each row
        valid_idx = 0
        for _, row in df.iterrows():
            # Try to process both images first to check if they exist
            try:
                claim_img = Image.open(row['claim_image']).convert('RGB')
                claim_img_tensor = preprocess(claim_img).numpy()
                
                doc_img = Image.open(row['evidence_image']).convert('RGB')
                doc_img_tensor = preprocess(doc_img).numpy()
            except Exception as e:
                logger.warning(f"Skipping sample due to missing image: {e}")
                continue
                
            # Create group for this sample only if both images are valid
            sample_group = f.create_group(str(valid_idx))
            
            # Store text data
            sample_group.create_dataset('claim', data=row['claim_enriched' if enriched else 'claim'])
            sample_group.create_dataset('document', data=row['evidence_enriched' if enriched else 'evidence'])
            
            # Store the processed images
            sample_group.create_dataset('claim_image', data=claim_img_tensor)
            sample_group.create_dataset('document_image', data=doc_img_tensor)
            
            # Store multi-path labels
            labels = category_to_labels.get(row['category'], [1, 1, 1, 1])  # Default to NEI if category not found
            sample_group.create_dataset('labels', data=np.array(labels, dtype=np.int64))
            
            valid_idx += 1
    
    logger.info(f"Created H5 dataset at {h5_path} with {valid_idx} valid samples")


class MisinformationDataset(Dataset):
    def __init__(self, csv_path, pre_embed=False):
        self.csv_path = csv_path
        self.pre_embed = pre_embed
        
        # Derive h5 path from csv path
        base_path = os.path.splitext(csv_path)[0]
        self.h5_path = base_path + '_embeddings.h5' if pre_embed else base_path + '.h5'
        
        if not os.path.exists(self.h5_path):
            if pre_embed:
                raise FileNotFoundError(f"Pre-computed embeddings not found at {self.h5_path}. "
                                      f"Please run preprocess_embeddings.py first.")
            logger.info(f"H5 file not found at {self.h5_path}. Creating new H5 dataset...")
            prepare_h5_dataset(self.csv_path, self.h5_path)
        
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.length = len(self.h5_file.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.h5_file[str(idx)]
        
        if self.pre_embed:
            return {
                'id': str(idx),
                'claim_text_embeds': torch.from_numpy(sample['claim_text_embeds'][()]),
                'doc_text_embeds': torch.from_numpy(sample['doc_text_embeds'][()]),
                'claim_image_embeds': torch.from_numpy(sample['claim_image_embeds'][()]),
                'doc_image_embeds': torch.from_numpy(sample['doc_image_embeds'][()]),
                'labels': torch.from_numpy(sample['labels'][()])
            }
        else:
            return {
                'id': str(idx),
                'claim': sample['claim'][()].decode(),
                'claim_image': torch.from_numpy(sample['claim_image'][()]),
                'document': sample['document'][()].decode(),
                'document_image': torch.from_numpy(sample['document_image'][()]),
                'labels': torch.from_numpy(sample['labels'][()])
            }

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def get_dataloader(csv_path, batch_size=32, num_workers=4, shuffle=False, pre_embed=False):
    dataset = MisinformationDataset(csv_path, pre_embed=pre_embed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloaders
    train_loader = get_dataloader('data/preprocessed/train.csv', shuffle=True)
    #test_loader = get_dataloader('data/preprocessed/test.csv', shuffle=False)
    
    # Test dataloaders
    for batch in train_loader:
        print("Train batch:")
        print(f"Batch size: {len(batch['id'])}")
        print(f"Claim shape: {batch['claim_image'].shape}")
        print(f"Document image shape: {batch['document_image'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")  # Should be (batch_size, 4)
        print(f"Sample labels: {batch['labels'][0]}")  # Show labels for first item
        break

    #for batch in test_loader:
    #    print("\nTest batch:")
    #    print(f"Batch size: {len(batch['id'])}")
    #    print(f"Claim shape: {batch['claim_image'].shape}")
    #    print(f"Document image shape: {batch['document_image'].shape}")
    #    print(f"Labels shape: {batch['labels'].shape}")
    #    print(f"Sample labels: {batch['labels'][0]}")
    #    break