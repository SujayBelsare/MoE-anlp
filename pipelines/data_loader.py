import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Optional


class XSumDataset(Dataset):
    """XSum dataset for extreme summarization"""
    
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_target_length: int = 64,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            split: 'train', 'validation', or 'test'
            tokenizer: Tokenizer to use
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            num_samples: Number of samples to load (None for all)
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Load dataset
        print(f"Loading {split} split of XSum dataset...")
        self.dataset = load_dataset("EdinburghNLP/xsum", split=split)
        
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Tokenize document
        document_encoding = self.tokenizer(
            item['document'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize summary
        summary_encoding = self.tokenizer(
            item['summary'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': document_encoding['input_ids'].squeeze(0),
            'attention_mask': document_encoding['attention_mask'].squeeze(0),
            'labels': summary_encoding['input_ids'].squeeze(0),
            'decoder_attention_mask': summary_encoding['attention_mask'].squeeze(0),
            'document': item['document'],
            'summary': item['summary'],
        }


def get_data_loader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    split: str,
    max_input_length: int = 512,
    max_target_length: int = 64,
    num_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """
    Factory function to create a DataLoader for XSum dataset
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size
        split: 'train', 'validation', or 'test'
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        num_samples: Number of samples to load
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (None = auto based on split)
    
    Returns:
        DataLoader instance
    """
    dataset = XSumDataset(
        split=split,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        num_samples=num_samples,
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class InstructDataset(Dataset):
    """Dataset wrapper for instruction-tuning format"""
    
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        print(f"Loading {split} split for instruction tuning...")
        self.dataset = load_dataset("EdinburghNLP/xsum", split=split)
        
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
        
        # Instruction template
        self.instruction = (
            "Summarize the following news article in one sentence:\n\n"
            "{document}\n\nSummary:"
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Format as instruction
        prompt = self.instruction.format(document=item['document'])
        full_text = prompt + " " + item['summary']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (mask the prompt part)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_len] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'document': item['document'],
            'summary': item['summary'],
        }


def get_instruct_data_loader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    split: str,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """Create DataLoader for instruction-tuning"""
    dataset = InstructDataset(
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_samples,
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )