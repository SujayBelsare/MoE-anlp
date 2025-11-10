"""
Comprehensive evaluation script for summarization models
Includes lexical, embedding-based, and factual consistency metrics
Configuration is loaded from config.yaml
"""

import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu import corpus_bleu
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import json
import yaml
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SummarizationEvaluator:
    """Comprehensive evaluator for summarization models"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.stop_words = set(stopwords.words('english'))
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        print("Computing ROUGE scores...")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': {
                'mean': np.mean(rouge1_scores),
                'std': np.std(rouge1_scores)
            },
            'rouge2': {
                'mean': np.mean(rouge2_scores),
                'std': np.std(rouge2_scores)
            },
            'rougeL': {
                'mean': np.mean(rougeL_scores),
                'std': np.std(rougeL_scores)
            }
        }
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute BLEU score
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with BLEU score
        """
        print("Computing BLEU score...")
        
        # Format references for sacrebleu (needs list of lists)
        references_formatted = [[ref] for ref in references]
        
        bleu = corpus_bleu(predictions, references_formatted)
        
        return {
            'bleu': {
                'score': bleu.score,
                'precisions': bleu.precisions
            }
        }
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 32,
        device: str = None
    ) -> Dict:
        """
        Compute BERTScore
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            batch_size: Batch size for processing
            device: Device to use (cuda/cpu)
            
        Returns:
            Dictionary with BERTScore
        """
        print("Computing BERTScore...")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        P, R, F1 = bert_score(
            predictions,
            references,
            lang="en",
            batch_size=batch_size,
            device=device,
            verbose=True
        )
        
        return {
            'bertscore': {
                'precision': {
                    'mean': P.mean().item(),
                    'std': P.std().item()
                },
                'recall': {
                    'mean': R.mean().item(),
                    'std': R.std().item()
                },
                'f1': {
                    'mean': F1.mean().item(),
                    'std': F1.std().item()
                }
            }
        }
    
    def compute_compression_ratio(
        self,
        predictions: List[str],
        documents: List[str]
    ) -> Dict:
        """
        Compute compression ratio (summary length / document length)
        
        Args:
            predictions: List of predicted summaries
            documents: List of source documents
            
        Returns:
            Dictionary with compression ratios
        """
        print("Computing compression ratios...")
        
        ratios = []
        for pred, doc in zip(predictions, documents):
            pred_words = len(pred.split())
            doc_words = len(doc.split())
            if doc_words > 0:
                ratios.append(pred_words / doc_words)
        
        return {
            'compression_ratio': {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'min': np.min(ratios),
                'max': np.max(ratios)
            }
        }
    
    def compute_extractiveness(
        self,
        predictions: List[str],
        documents: List[str]
    ) -> Dict:
        """
        Compute extractiveness: percentage of content words in summary that appear in document
        
        Args:
            predictions: List of predicted summaries
            documents: List of source documents
            
        Returns:
            Dictionary with extractiveness scores
        """
        print("Computing extractiveness...")
        
        extractiveness_scores = []
        
        for pred, doc in tqdm(zip(predictions, documents), total=len(predictions)):
            # Tokenize and remove stopwords
            pred_tokens = set([
                w.lower() for w in word_tokenize(pred)
                if w.lower() not in self.stop_words and w.isalnum()
            ])
            
            doc_tokens = set([
                w.lower() for w in word_tokenize(doc)
                if w.lower() not in self.stop_words and w.isalnum()
            ])
            
            if len(pred_tokens) > 0:
                overlap = len(pred_tokens.intersection(doc_tokens))
                extractiveness = overlap / len(pred_tokens)
                extractiveness_scores.append(extractiveness)
        
        return {
            'extractiveness': {
                'mean': np.mean(extractiveness_scores),
                'std': np.std(extractiveness_scores),
                'min': np.min(extractiveness_scores),
                'max': np.max(extractiveness_scores)
            }
        }
        
    def human_evaluation_samples(
        self,
        predictions: List[str],
        references: List[str],
        documents: List[str],
        num_samples: int = 3
    ) -> List[Dict]:
        """
        Select samples for human evaluation
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            documents: List of source documents
            num_samples: Number of samples to select
            
        Returns:
            List of sample dictionaries
        """
        print(f"\nSelecting {num_samples} samples for human evaluation...")
        
        # Select diverse samples (beginning, middle, end)
        indices = np.linspace(0, len(predictions) - 1, num_samples, dtype=int)
        
        samples = []
        for idx in indices:
            samples.append({
                'index': int(idx),
                'document': documents[idx],
                'reference': references[idx],
                'prediction': predictions[idx]
            })
        
        return samples
    
    def print_human_eval_samples(self, samples: List[Dict]):
        """
        Print samples for manual human evaluation
        
        Dimensions to evaluate:
        - Content Relevance
        - Coherence
        - Fluency
        - Factual Consistency
        """
        print("\n" + "="*80)
        print("SAMPLES FOR HUMAN EVALUATION")
        print("="*80)
        print("\nPlease evaluate each sample on the following dimensions:")
        print("1. Content Relevance: Does the summary cover the main points?")
        print("2. Coherence: Does the summary flow smoothly and make sense?")
        print("3. Fluency: Is the summary grammatically correct?")
        print("4. Factual Consistency: Is the information in the summary faithful to the document?")
        print("\n" + "="*80)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*80}")
            print(f"SAMPLE {i} (Index: {sample['index']})")
            print(f"{'='*80}")
            
            print(f"\n[DOCUMENT]")
            print(sample['document'][:500] + "..." if len(sample['document']) > 500 else sample['document'])
            
            print(f"\n[REFERENCE SUMMARY]")
            print(sample['reference'])
            
            print(f"\n[PREDICTED SUMMARY]")
            print(sample['prediction'])
            
            print(f"\n[EVALUATION NOTES]")
            print("Content Relevance (1-5): ___")
            print("Coherence (1-5): ___")
            print("Fluency (1-5): ___")
            print("Factual Consistency (1-5): ___")
            print("Comments: ________________________________")
        
        print("\n" + "="*80)
    
    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str],
        documents: List[str],
        human_eval_samples: int = 3
    ) -> Dict:
        """
        Run all evaluation metrics
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            documents: List of source documents
            human_eval_samples: Number of samples for human evaluation
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Lexical metrics
        print("\n=== LEXICAL METRICS ===")
        results.update(self.compute_rouge(predictions, references))
        results.update(self.compute_bleu(predictions, references))
        
        # Embedding-based metrics
        print("\n=== EMBEDDING-BASED METRICS ===")
        results.update(self.compute_bertscore(predictions, references))
        
        # Document-related metrics
        print("\n=== DOCUMENT-RELATED METRICS ===")
        results.update(self.compute_compression_ratio(predictions, documents))
        results.update(self.compute_extractiveness(predictions, documents))
        
        
        # Human evaluation samples
        if human_eval_samples > 0:
            samples = self.human_evaluation_samples(
                predictions,
                references,
                documents,
                num_samples=human_eval_samples
            )
            results['human_eval_samples'] = samples
            self.print_human_eval_samples(samples)
        
        return results


def load_predictions(file_path: str) -> tuple:
    """Load predictions from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data['predictions'], data['references'], data.get('documents', [])


def print_results(results: Dict, model_name: str):
    """Print evaluation results in a readable format"""
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*80)
    
    # ROUGE scores
    if 'rouge1' in results:
        print("\n--- ROUGE Scores ---")
        print(f"ROUGE-1: {results['rouge1']['mean']:.4f} ± {results['rouge1']['std']:.4f}")
        print(f"ROUGE-2: {results['rouge2']['mean']:.4f} ± {results['rouge2']['std']:.4f}")
        print(f"ROUGE-L: {results['rougeL']['mean']:.4f} ± {results['rougeL']['std']:.4f}")
    
    # BLEU score
    if 'bleu' in results:
        print("\n--- BLEU Score ---")
        print(f"BLEU: {results['bleu']['score']:.4f}")
    
    # BERTScore
    if 'bertscore' in results:
        print("\n--- BERTScore ---")
        print(f"Precision: {results['bertscore']['precision']['mean']:.4f} ± {results['bertscore']['precision']['std']:.4f}")
        print(f"Recall: {results['bertscore']['recall']['mean']:.4f} ± {results['bertscore']['recall']['std']:.4f}")
        print(f"F1: {results['bertscore']['f1']['mean']:.4f} ± {results['bertscore']['f1']['std']:.4f}")
    
    # Compression ratio
    if 'compression_ratio' in results:
        print("\n--- Compression Ratio ---")
        print(f"Mean: {results['compression_ratio']['mean']:.4f} ± {results['compression_ratio']['std']:.4f}")
        print(f"Range: [{results['compression_ratio']['min']:.4f}, {results['compression_ratio']['max']:.4f}]")
    
    # Extractiveness
    if 'extractiveness' in results:
        print("\n--- Extractiveness ---")
        print(f"Mean: {results['extractiveness']['mean']:.4f} ± {results['extractiveness']['std']:.4f}")
        print(f"Range: [{results['extractiveness']['min']:.4f}, {results['extractiveness']['max']:.4f}]")
        
    print("\n" + "="*80)


def compare_models(results_dict: Dict[str, Dict]):
    """Compare multiple models side by side"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Create comparison table
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bertscore', 
               'compression_ratio', 'extractiveness']
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {'Model': model_name}
        
        if 'rouge1' in results:
            row['ROUGE-1'] = f"{results['rouge1']['mean']:.4f}"
            row['ROUGE-2'] = f"{results['rouge2']['mean']:.4f}"
            row['ROUGE-L'] = f"{results['rougeL']['mean']:.4f}"
        
        if 'bleu' in results:
            row['BLEU'] = f"{results['bleu']['score']:.4f}"
        
        if 'bertscore' in results:
            row['BERTScore-F1'] = f"{results['bertscore']['f1']['mean']:.4f}"
        
        if 'compression_ratio' in results:
            row['Compression'] = f"{results['compression_ratio']['mean']:.4f}"
        
        if 'extractiveness' in results:
            row['Extractiveness'] = f"{results['extractiveness']['mean']:.4f}"
        
        comparison_data.append(row)
    
    # Print as DataFrame
    df = pd.DataFrame(comparison_data)
    print("\n", df.to_string(index=False))
    print("\n" + "="*80)
    
    return df


def main():
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    eval_config = config['scoring']
    
    # Load predictions
    print(f"Loading predictions from {eval_config['predictions_file']}")
    predictions, references, documents = load_predictions(eval_config['predictions_file'])
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Create evaluator
    evaluator = SummarizationEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all(
        predictions=predictions,
        references=references,
        documents=documents,
        human_eval_samples=eval_config['human_eval_samples']
    )
    
    # Print results
    print_results(results, eval_config['model_name'])
    
    # Save results
    # Remove human eval samples from saved results (they're just for display)
    results_to_save = {k: v for k, v in results.items() if k != 'human_eval_samples'}
    
    with open(eval_config['output_file'], 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to {eval_config['output_file']}")
    
    # Compare with other models if specified
    if eval_config['compare_with']:
        print("\n" + "="*80)
        print("LOADING ADDITIONAL MODELS FOR COMPARISON")
        print("="*80)
        
        all_results = {eval_config['model_name']: results_to_save}
        
        for comp_file in eval_config['compare_with']:
            comp_name = comp_file.split('/')[-1].replace('_predictions.json', '')
            print(f"\nEvaluating {comp_name}...")
            
            comp_preds, comp_refs, comp_docs = load_predictions(comp_file)
            comp_results = evaluator.evaluate_all(
                predictions=comp_preds,
                references=comp_refs,
                documents=comp_docs,
                human_eval_samples=0  # Don't print samples for comparison models
            )
            
            all_results[comp_name] = {k: v for k, v in comp_results.items() 
                                     if k != 'human_eval_samples'}
        
        # Create comparison table
        comparison_df = compare_models(all_results)
        
        # Save comparison
        comparison_output = eval_config['output_file'].replace('.json', '_comparison.csv')
        comparison_df.to_csv(comparison_output, index=False)
        print(f"\nComparison saved to {comparison_output}")


if __name__ == "__main__":
    main()