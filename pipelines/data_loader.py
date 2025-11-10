"""
Data loading and preprocessing for XSum dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import config


class XSumDataset(Dataset):
    """
    PyTorch Dataset for XSum extreme summarization task.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer,
        max_source_length: int = config.MAX_SOURCE_LENGTH,
        max_target_length: int = config.MAX_TARGET_LENGTH,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer instance
            max_source_length: Maximum length for source documents
            max_target_length: Maximum length for target summaries
            num_samples: Optional limit on number of samples (for debugging)
        """
        self.split = split
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load dataset from HuggingFace
        print(f"Loading XSum dataset split: {split}")
        self.dataset = load_dataset(config.DATASET_NAME, split=split)
        
        # Optionally limit dataset size
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
                - input_ids: Tokenized source document
                - attention_mask: Attention mask for source
                - labels: Tokenized target summary
                - decoder_attention_mask: Attention mask for target
        """
        sample = self.dataset[idx]
        
        # Get document and summary
        document = sample['document']
        summary = sample['summary']
        
        # Tokenize source document
        source_encoding = self.tokenizer(
            document,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100 for loss calculation)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }


class XSumDataModule:
    """
    Data module that handles dataset creation and dataloader initialization.
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        batch_size: int = config.BATCH_SIZE,
        max_source_length: int = config.MAX_SOURCE_LENGTH,
        max_target_length: int = config.MAX_TARGET_LENGTH,
        num_workers: int = 4,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None
    ):
        """
        Args:
            tokenizer_name: Name or path of tokenizer
            batch_size: Batch size for dataloaders
            max_source_length: Maximum source document length
            max_target_length: Maximum target summary length
            num_workers: Number of dataloader workers
            train_samples: Optional limit on training samples
            val_samples: Optional limit on validation samples
            test_samples: Optional limit on test samples
        """
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create datasets
        self.train_dataset = XSumDataset(
            'train',
            self.tokenizer,
            max_source_length,
            max_target_length,
            train_samples
        )
        
        self.val_dataset = XSumDataset(
            'validation',
            self.tokenizer,
            max_source_length,
            max_target_length,
            val_samples
        )
        
        self.test_dataset = XSumDataset(
            'test',
            self.tokenizer,
            max_source_length,
            max_target_length,
            test_samples
        )
    
    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_tokenizer(self):
        """Get the tokenizer instance"""
        return self.tokenizer


def get_data_loader(
    tokenizer,
    split: str,
    batch_size: int = config.BATCH_SIZE,
    max_source_length: int = config.MAX_SOURCE_LENGTH,
    max_target_length: int = config.MAX_TARGET_LENGTH,
    shuffle: bool = None,
    num_workers: int = 4,
    num_samples: Optional[int] = None
) -> DataLoader:
    """
    Factory function to create a dataloader for a specific split.
    
    Args:
        tokenizer: Tokenizer instance
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        max_source_length: Maximum source length
        max_target_length: Maximum target length
        shuffle: Whether to shuffle (default: True for train, False for val/test)
        num_workers: Number of dataloader workers
        num_samples: Optional limit on number of samples
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = XSumDataset(
        split=split,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        num_samples=num_samples
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Collate function for variable length sequences
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    # Stack all tensors
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'decoder_attention_mask': torch.stack([item['decoder_attention_mask'] for item in batch])
    }