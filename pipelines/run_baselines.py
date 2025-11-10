"""
Script to run baseline models: BART inference, Finetune encoder-decoder, Instruction tune
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
from tqdm import tqdm
import json
from config import load_config, DEFAULT_CONFIG as config
from utils import set_seed, ensure_dir


class BARTInference:
    """Run inference with pre-trained BART model"""
    
    def __init__(self, model_name: str = config.BART_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BART model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate_summaries(
        self,
        documents,
        max_length: int = config.MAX_TARGET_LENGTH,
        num_beams: int = config.NUM_BEAMS,
        batch_size: int = 8
    ):
        """Generate summaries for a list of documents"""
        summaries = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Generating summaries"):
            batch_docs = documents[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_docs,
                max_length=config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=config.LENGTH_PENALTY,
                early_stopping=config.EARLY_STOPPING
            )
            
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def run_on_test_set(self, output_path: str = "outputs/bart_predictions.json"):
        """Run inference on XSum test set"""
        print("Loading XSum test set...")
        dataset = load_dataset(config.DATASET_NAME, split="test")
        
        documents = dataset['document']
        references = dataset['summary']
        
        print(f"Generating summaries for {len(documents)} documents...")
        predictions = self.generate_summaries(documents)
        
        # Save results
        ensure_dir(os.path.dirname(output_path))
        results = {
            'predictions': predictions,
            'references': references,
            'documents': documents
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return predictions, references


class EncoderDecoderFineTuner:
    """Fine-tune encoder-decoder models (T5, Pegasus)"""
    
    def __init__(
        self,
        model_name: str,
        use_peft: bool = True,
        use_8bit: bool = False,
        device: str = None
    ):
        self.model_name = model_name
        self.use_peft = use_peft
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization if needed
        if use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Apply PEFT if specified
        if use_peft:
            print("Applying LoRA...")
            self.model = self._apply_lora(self.model)
        
        # Move to device if not using device_map
        if not use_8bit:
            self.model.to(self.device)
    
    def _apply_lora(self, model):
        """Apply LoRA to the model"""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]  # Target attention matrices
        )
        
        if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def prepare_dataset(self, split: str, num_samples: int = None):
        """Prepare dataset for training/evaluation"""
        dataset = load_dataset(config.DATASET_NAME, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        def preprocess(examples):
            inputs = self.tokenizer(
                examples['document'],
                max_length=config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding=False
            )
            
            targets = self.tokenizer(
                examples['summary'],
                max_length=config.MAX_TARGET_LENGTH,
                truncation=True,
                padding=False
            )
            
            inputs['labels'] = targets['input_ids']
            return inputs
        
        processed_dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def train(
        self,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        train_samples: int = None,
        val_samples: int = None,
        push_to_hub: bool = False
    ):
        """Fine-tune the model"""
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset("train", train_samples)
        val_dataset = self.prepare_dataset("validation", val_samples)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            predict_with_generate=True,
            generation_max_length=config.MAX_TARGET_LENGTH,
            generation_num_beams=config.NUM_BEAMS,
            push_to_hub=push_to_hub,
            report_to="wandb" if not os.getenv("NO_WANDB") else "none"
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")
        
        return trainer
    
    def inference(
        self,
        documents,
        batch_size: int = 8,
        max_length: int = config.MAX_TARGET_LENGTH
    ):
        """Run inference on documents"""
        self.model.eval()
        summaries = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Generating"):
            batch_docs = documents[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_docs,
                max_length=config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=config.NUM_BEAMS,
                    length_penalty=config.LENGTH_PENALTY,
                    early_stopping=config.EARLY_STOPPING
                )
            
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        return summaries


class InstructionTuner:
    """Instruction tune LLaMA or Qwen models"""
    
    def __init__(
        self,
        model_name: str,
        use_peft: bool = True,
        use_4bit: bool = True,
        device: str = None
    ):
        self.model_name = model_name
        self.use_peft = use_peft
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading instruction model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Configure quantization
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to(self.device)
        
        # Apply PEFT
        if use_peft:
            print("Applying LoRA...")
            self.model = self._apply_lora(self.model)
    
    def _apply_lora(self, model):
        """Apply LoRA for instruction tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def format_prompt(self, document: str, summary: str = None):
        """Format document and summary as instruction prompt"""
        instruction = "Summarize the following article in one sentence:"
        
        if summary is None:
            prompt = f"<|system|>\nYou are a helpful assistant that summarizes news articles.\n<|user|>\n{instruction}\n\n{document}\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\nYou are a helpful assistant that summarizes news articles.\n<|user|>\n{instruction}\n\n{document}\n<|assistant|>\n{summary}"
        
        return prompt
    
    def prepare_dataset(self, split: str, num_samples: int = None):
        """Prepare dataset for instruction tuning"""
        dataset = load_dataset(config.DATASET_NAME, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        def preprocess(examples):
            prompts = []
            for doc, summ in zip(examples['document'], examples['summary']):
                prompts.append(self.format_prompt(doc, summ))
            
            tokenized = self.tokenizer(
                prompts,
                max_length=config.MAX_SOURCE_LENGTH + config.MAX_TARGET_LENGTH,
                truncation=True,
                padding=False
            )
            
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        processed_dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def train(
        self,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        train_samples: int = None,
        val_samples: int = None,
        push_to_hub: bool = False
    ):
        """Instruction tune the model"""
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset("train", train_samples)
        val_dataset = self.prepare_dataset("validation", val_samples)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=push_to_hub,
            fp16=True,
            report_to="wandb" if not os.getenv("NO_WANDB") else "none"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save
        trainer.save_model(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")
        
        return trainer
    
    def inference(
        self,
        documents,
        batch_size: int = 4,
        max_new_tokens: int = config.MAX_TARGET_LENGTH
    ):
        """Generate summaries"""
        self.model.eval()
        summaries = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Generating"):
            batch_docs = documents[i:i+batch_size]
            prompts = [self.format_prompt(doc) for doc in batch_docs]
            
            inputs = self.tokenizer(
                prompts,
                max_length=config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=config.NUM_BEAMS,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract summary (after the prompt)
            for j, output in enumerate(outputs):
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Extract only the summary part after <|assistant|>
                if "<|assistant|>" in full_text:
                    summary = full_text.split("<|assistant|>")[-1].strip()
                else:
                    summary = full_text
                summaries.append(summary)
        
        return summaries


def main():
    # Load configuration
    config = load_config()
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Debug mode
    if config["mode"]["debug"]:
        config["dataset"]["train_samples"] = 100
        config["dataset"]["val_samples"] = 50
        config["training"]["epochs"] = 1
        print("DEBUG MODE: Using small dataset")
    
    # Run task
    if config["mode"]["task"] == "bart_inference":
        print("Running BART inference...")
        bart = BARTInference()
        predictions, references = bart.run_on_test_set(
            output_path=f"{config['paths']['outputs']}/bart_predictions.json"
        )
        print(f"Generated {len(predictions)} summaries")
    
    elif config["mode"]["task"] == "finetune":
        model = config["model"].get("name", "google-t5/t5-base")
        print(f"Fine-tuning {model}...")
        finetuner = EncoderDecoderFineTuner(
            model_name=model,
            use_peft=config["training"]["use_peft"],
            use_8bit=config["training"].get("use_quantization", False)
        )
        
        output_dir = f"{config['paths']['outputs']}/finetuned_{model.split('/')[-1]}"
        
        trainer = finetuner.train(
            output_dir=output_dir,
            num_epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            train_samples=config["dataset"]["train_samples"],
            val_samples=config["dataset"]["val_samples"],
            push_to_hub=config["huggingface"]["push_to_hub"]
        )
        
        # Run inference on test set
        print("Running inference on test set...")
        test_dataset = load_dataset(config.DATASET_NAME, split="test")
        predictions = finetuner.inference(test_dataset['document'])
        
        results = {
            'predictions': predictions,
            'references': test_dataset['summary'],
            'documents': test_dataset['document']
        }
        
        result_path = f"{output_dir}/test_predictions.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_path}")
    
    elif config["mode"]["task"] == "instruction_tune":
        model = config["model"].get("name", "meta-llama/Llama-3.2-1B-Instruct")
        
        print(f"Instruction tuning {model}...")
        tuner = InstructionTuner(
            model_name=model,
            use_peft=config["training"]["use_peft"],
            use_4bit=config["training"].get("use_quantization", False)
        )
        
        output_dir = f"{config['paths']['outputs']}/instruction_tuned_{model.split('/')[-1]}"
        
        trainer = tuner.train(
            output_dir=output_dir,
            num_epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            train_samples=config["dataset"]["train_samples"],
            val_samples=config["dataset"]["val_samples"],
            push_to_hub=config["huggingface"]["push_to_hub"]
        )
        
        # Run inference on test set
        print("Running inference on test set...")
        test_dataset = load_dataset(config.DATASET_NAME, split="test")
        predictions = tuner.inference(test_dataset['document'])
        
        results = {
            'predictions': predictions,
            'references': test_dataset['summary'],
            'documents': test_dataset['document']
        }
        
        result_path = f"{output_dir}/test_predictions.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()