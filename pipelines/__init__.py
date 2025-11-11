# pipelines/__init__.py
"""
Pipelines module for ANLP Assignment 3
Contains data loading, training, and evaluation pipelines
"""

from .data_loader import XSumDataset, get_data_loader, InstructDataset, get_instruct_data_loader
from .run_baselines import run_bart_inference, finetune_encoder_decoder, instruction_tune_model
from .train_moe import MoETrainer, train_moe_model
from .evaluate import (
    compute_lexical_metrics,
    compute_embedding_metrics,
    compute_doc_metrics,
    get_human_eval_samples,
    evaluate_moe_model,
    evaluate_baseline
)

__all__ = [
    'XSumDataset',
    'get_data_loader',
    'InstructDataset',
    'get_instruct_data_loader',
    'run_bart_inference',
    'finetune_encoder_decoder',
    'instruction_tune_model',
    'MoETrainer',
    'train_moe_model',
    'compute_lexical_metrics',
    'compute_embedding_metrics',
    'compute_doc_metrics',
    'get_human_eval_samples',
    'evaluate_moe_model',
    'evaluate_baseline',
]