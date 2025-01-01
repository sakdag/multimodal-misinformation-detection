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

def prepare_h5_dataset(csv_path, h5_path):
    """
    Prepare h5 dataset from CSV file where each index contains complete sample data
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path, index_col=0)[['claim', 'claim_image', 'evidence', 'evidence_image', 'category']]
    
    with h5py.File(h5_path, 'w') as f:
        # Process each row
        for idx, (_, row) in enumerate(df.iterrows()):
            # Create group for this sample
            sample_group = f.create_group(str(idx))
            
            # Store text data
            sample_group.create_dataset('claim', data=row['claim'])
            sample_group.create_dataset('document', data=row['evidence'])
            
            # Process and store images
            try:
                claim_img = Image.open(row['claim_image']).convert('RGB')
                claim_img_tensor = preprocess(claim_img).numpy()
            except Exception as e:
                logger.warning(f"Error processing claim image for idx {idx}: {e}")
                claim_img_tensor = np.zeros((3, 256, 256), dtype='float32')
            sample_group.create_dataset('claim_image', data=claim_img_tensor)
            
            try:
                doc_img = Image.open(row['evidence_image']).convert('RGB')
                doc_img_tensor = preprocess(doc_img).numpy()
            except Exception as e:
                logger.warning(f"Error processing evidence image for idx {idx}: {e}")
                doc_img_tensor = np.zeros((3, 256, 256), dtype='float32')
            sample_group.create_dataset('document_image', data=doc_img_tensor)
            
            # Store multi-path labels
            labels = category_to_labels.get(row['category'], [1, 1, 1, 1])  # Default to NEI if category not found
            sample_group.create_dataset('labels', data=np.array(labels, dtype=np.int64))
    
    logger.info(f"Created H5 dataset at {h5_path}")


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