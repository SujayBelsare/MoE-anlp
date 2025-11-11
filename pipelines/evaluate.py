import torch
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Dict, List
import json
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import MoETransformer
from pipelines.data_loader import get_data_loader

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def compute_lexical_metrics(summaries: List[str], references: List[str]) -> Dict:
    """Compute ROUGE-N, ROUGE-L, and BLEU scores"""
    print("Computing lexical metrics...")
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for summary, reference in zip(summaries, references):
        scores = scorer.score(reference, summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    for summary, reference in zip(summaries, references):
        summary_tokens = nltk.word_tokenize(summary.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        
        try:
            bleu = sentence_bleu(reference_tokens, summary_tokens, 
                               smoothing_function=smoothie)
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
    
    return {
        'rouge1': {
            'mean': np.mean(rouge1_scores),
            'std': np.std(rouge1_scores),
        },
        'rouge2': {
            'mean': np.mean(rouge2_scores),
            'std': np.std(rouge2_scores),
        },
        'rougeL': {
            'mean': np.mean(rougeL_scores),
            'std': np.std(rougeL_scores),
        },
        'bleu': {
            'mean': np.mean(bleu_scores),
            'std': np.std(bleu_scores),
        },
    }


def compute_embedding_metrics(summaries: List[str], references: List[str]) -> Dict:
    """Compute BERTScore"""
    print("Computing BERTScore...")
    
    P, R, F1 = bert_score(summaries, references, lang='en', verbose=False)
    
    return {
        'precision': {
            'mean': P.mean().item(),
            'std': P.std().item(),
        },
        'recall': {
            'mean': R.mean().item(),
            'std': R.std().item(),
        },
        'f1': {
            'mean': F1.mean().item(),
            'std': F1.std().item(),
        },
    }


def compute_doc_metrics(summaries: List[str], documents: List[str]) -> Dict:
    """Compute Compression Ratio and Extractiveness"""
    print("Computing document metrics...")
    
    compression_ratios = []
    extractiveness_scores = []
    
    # Load stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    for summary, document in zip(summaries, documents):
        # Compression ratio
        summary_len = len(summary.split())
        document_len = len(document.split())
        compression_ratio = summary_len / document_len if document_len > 0 else 0
        compression_ratios.append(compression_ratio)
        
        # Extractiveness
        summary_words = set([w.lower() for w in nltk.word_tokenize(summary) 
                           if w.lower() not in stop_words and w.isalnum()])
        document_words = set([w.lower() for w in nltk.word_tokenize(document) 
                            if w.lower() not in stop_words and w.isalnum()])
        
        if len(summary_words) > 0:
            overlap = len(summary_words.intersection(document_words))
            extractiveness = overlap / len(summary_words)
        else:
            extractiveness = 0.0
        
        extractiveness_scores.append(extractiveness)
    
    return {
        'compression_ratio': {
            'mean': np.mean(compression_ratios),
            'std': np.std(compression_ratios),
        },
        'extractiveness': {
            'mean': np.mean(extractiveness_scores),
            'std': np.std(extractiveness_scores),
        },
    }


def get_human_eval_samples(
    summaries: List[str],
    documents: List[str],
    references: List[str],
    num_samples: int = 3
) -> List[Dict]:
    """Get samples for human evaluation"""
    print(f"Selecting {num_samples} samples for human evaluation...")
    
    # Randomly select samples
    indices = np.random.choice(len(summaries), num_samples, replace=False)
    
    samples = []
    for idx in indices:
        samples.append({
            'document': documents[idx],
            'reference': references[idx],
            'generated': summaries[idx],
            'metrics_to_evaluate': [
                'Content Relevance (1-5): How well does the summary cover main points?',
                'Coherence (1-5): Does the summary flow logically?',
                'Fluency (1-5): Grammar and language quality?',
                'Factual Consistency (1-5): Is information faithful to source?',
            ]
        })
    
    return samples


def evaluate_moe_model(config: Dict, router_type: str, use_load_balancer: bool = False):
    """Evaluate a trained MoE model"""
    print(f"\nEvaluating MoE model with {router_type} routing...")
    
    # Setup
    lb_suffix = "_with_lb" if use_load_balancer else "_no_lb"
    model_name = f"moe_{router_type}{lb_suffix}"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_dir = os.path.join(config['output']['model_dir'], model_name, 'best')
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = MoETransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        num_experts=config['model']['num_experts'],
        expert_hidden_dim=config['model']['expert_hidden_dim'],
        top_k=config['model']['top_k'],
        router_type=router_type,
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_seq_length'],
        use_load_balancer=use_load_balancer,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load test data
    test_loader = get_data_loader(
        tokenizer,
        config['evaluation']['batch_size'],
        'test',
        config['data']['max_input_length'],
        config['data']['max_target_length'],
        config['data']['test_samples'],
        0,  # No multiprocessing for generation
        shuffle=False,
    )
    
    # Generate summaries
    print("Generating summaries...")
    summaries = []
    references = []
    documents = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            src = batch['input_ids'].to(device)
            src_key_padding_mask = (src == tokenizer.pad_token_id)
            
            # Encode
            memory, _ = model.encode(src, src_key_padding_mask=src_key_padding_mask)
            
            # Generate (greedy decoding)
            batch_size = src.size(0)
            generated = torch.full(
                (batch_size, 1),
                tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
                dtype=torch.long,
                device=device
            )
            
            for _ in range(config['evaluation']['max_length']):
                tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
                logits, _ = model.decode(
                    generated, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            # Decode
            batch_summaries = tokenizer.batch_decode(generated, skip_special_tokens=True)
            summaries.extend(batch_summaries)
            references.extend(batch['summary'])
            documents.extend(batch['document'])
    
    # Compute metrics
    results = {
        'model': model_name,
        'router_type': router_type,
        'use_load_balancer': use_load_balancer,
        'num_samples': len(summaries),
    }
    
    # Lexical metrics
    results['lexical_metrics'] = compute_lexical_metrics(summaries, references)
    
    # Embedding metrics
    results['embedding_metrics'] = compute_embedding_metrics(summaries, references)
    
    # Document metrics
    results['document_metrics'] = compute_doc_metrics(summaries, documents)
    
    # Human evaluation samples
    results['human_eval_samples'] = get_human_eval_samples(summaries, documents, references)
    
    # Save results
    results_dir = os.path.join(config['output']['results_dir'], model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save all summaries
    with open(os.path.join(results_dir, 'summaries.json'), 'w') as f:
        json.dump({
            'summaries': summaries,
            'references': references,
            'documents': documents,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print("\nMetrics Summary:")
    print(f"  ROUGE-1: {results['lexical_metrics']['rouge1']['mean']:.4f}")
    print(f"  ROUGE-2: {results['lexical_metrics']['rouge2']['mean']:.4f}")
    print(f"  ROUGE-L: {results['lexical_metrics']['rougeL']['mean']:.4f}")
    print(f"  BLEU: {results['lexical_metrics']['bleu']['mean']:.4f}")
    print(f"  BERTScore F1: {results['embedding_metrics']['f1']['mean']:.4f}")
    print(f"  Compression Ratio: {results['document_metrics']['compression_ratio']['mean']:.4f}")
    print(f"  Extractiveness: {results['document_metrics']['extractiveness']['mean']:.4f}")
    
    return results


def evaluate_baseline(results_file: str, model_name: str, config: Dict):
    """Evaluate a baseline model from saved results"""
    print(f"\nEvaluating {model_name} baseline...")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    summaries = data['summaries']
    references = data['references']
    documents = data['documents']
    
    # Compute metrics
    results = {
        'model': model_name,
        'num_samples': len(summaries),
    }
    
    results['lexical_metrics'] = compute_lexical_metrics(summaries, references)
    results['embedding_metrics'] = compute_embedding_metrics(summaries, references)
    results['document_metrics'] = compute_doc_metrics(summaries, documents)
    results['human_eval_samples'] = get_human_eval_samples(summaries, documents, references)
    
    # Save results
    results_dir = os.path.join(config['output']['results_dir'], model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_dir}")
    
    return results