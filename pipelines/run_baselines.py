import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm
import json
import os
from typing import Dict, List


def run_bart_inference(config: Dict, output_dir: str):
    """Task 1: Run inference with pre-trained BART"""
    print("\n" + "="*50)
    print("Task 1: BART Inference")
    print("="*50)
    
    model_name = config['baselines']['bart']['model_name']
    batch_size = config['baselines']['bart']['batch_size']
    
    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset("EdinburghNLP/xsum", split="test")
    if config['data']['test_samples']:
        test_dataset = test_dataset.select(range(config['data']['test_samples']))
    
    # Generate summaries
    summaries = []
    references = []
    documents = []
    
    print("Generating summaries...")
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i:i+batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch['document'],
            max_length=config['data']['max_input_length'],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config['evaluation']['max_length'],
                min_length=config['evaluation']['min_length'],
                num_beams=config['evaluation']['num_beams'],
                length_penalty=config['evaluation']['length_penalty'],
                no_repeat_ngram_size=config['evaluation']['no_repeat_ngram_size'],
            )
        
        # Decode
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(batch_summaries)
        references.extend(batch['summary'])
        documents.extend(batch['document'])
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'model': model_name,
        'summaries': summaries,
        'references': references,
        'documents': documents,
    }
    
    output_file = os.path.join(output_dir, 'bart_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Generated {len(summaries)} summaries")
    
    return results


def finetune_encoder_decoder(config: Dict, output_dir: str):
    """Task 2: Fine-tune encoder-decoder model"""
    print("\n" + "="*50)
    print("Task 2: Fine-tuning Encoder-Decoder Model")
    print("="*50)
    
    model_name = config['baselines']['encoder_decoder']['model_name']
    use_peft = config['baselines']['encoder_decoder']['use_peft']
    
    # Load tokenizer and model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Apply LoRA if specified
    if use_peft:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config['baselines']['encoder_decoder']['lora_r'],
            lora_alpha=config['baselines']['encoder_decoder']['lora_alpha'],
            lora_dropout=config['baselines']['encoder_decoder']['lora_dropout'],
            target_modules=["q", "v"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset("EdinburghNLP/xsum", split="train")
    val_dataset = load_dataset("EdinburghNLP/xsum", split="validation")
    
    if config['data']['train_samples']:
        train_dataset = train_dataset.select(range(config['data']['train_samples']))
    if config['data']['val_samples']:
        val_dataset = val_dataset.select(range(config['data']['val_samples']))
    
    # Preprocess function
    def preprocess_function(examples):
        inputs = tokenizer(
            examples['document'],
            max_length=config['data']['max_input_length'],
            truncation=True,
            padding='max_length',
        )
        
        # Tokenize targets with padding
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                examples['summary'],
                max_length=config['data']['max_target_length'],
                truncation=True,
                padding='max_length',
            )
        
        # Replace padding token id with -100 so it's ignored in loss
        labels = targets['input_ids']
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels
        ]
        
        inputs['labels'] = labels
        return inputs
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    # Training arguments
    model_output_dir = os.path.join(output_dir, 'encoder_decoder_model')
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=config['baselines']['encoder_decoder']['num_epochs'],
        per_device_train_batch_size=config['baselines']['encoder_decoder']['batch_size'],
        per_device_eval_batch_size=config['baselines']['encoder_decoder']['batch_size'],
        learning_rate=config['baselines']['encoder_decoder']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        predict_with_generate=True,
        fp16=config['baselines']['encoder_decoder']['fp16'] and torch.cuda.is_available(),
        bf16=config['baselines']['encoder_decoder']['bf16'] and torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['baselines']['encoder_decoder']['max_grad_norm']
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")
    
    # Run inference on test set
    print("Running inference on test set...")
    test_dataset = load_dataset("EdinburghNLP/xsum", split="test")
    if config['data']['test_samples']:
        test_dataset = test_dataset.select(range(config['data']['test_samples']))
    
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    predictions = trainer.predict(test_dataset)
    
    # Decode predictions
    summaries = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    references = [test_dataset[i]['summary'] for i in range(len(test_dataset))]
    documents = [test_dataset[i]['document'] for i in range(len(test_dataset))]
    
    # Save results
    results = {
        'model': model_name,
        'summaries': summaries,
        'references': references,
        'documents': documents,
    }
    
    output_file = os.path.join(output_dir, 'encoder_decoder_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return results


def instruction_tune_model(config: Dict, output_dir: str):
    """Task 3: Instruction-tune a model"""
    print("\n" + "="*50)
    print("Task 3: Instruction Tuning")
    print("="*50)
    
    model_name = config['baselines']['instruct']['model_name']
    use_peft = config['baselines']['instruct']['use_peft']
    load_in_8bit = config['baselines']['instruct'].get('load_in_8bit', False)
    
    # Quantization config
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer and model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if load_in_8bit else None,
    )
    
    # Apply LoRA if specified
    if use_peft:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['baselines']['instruct']['lora_r'],
            lora_alpha=config['baselines']['instruct']['lora_alpha'],
            lora_dropout=config['baselines']['instruct']['lora_dropout'],
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset("EdinburghNLP/xsum", split="train")
    val_dataset = load_dataset("EdinburghNLP/xsum", split="validation")
    
    if config['data']['train_samples']:
        train_dataset = train_dataset.select(range(config['data']['train_samples']))
    if config['data']['val_samples']:
        val_dataset = val_dataset.select(range(config['data']['val_samples']))
    
    # Instruction template
    instruction_template = (
        "Summarize the following news article in one sentence:\n\n"
        "{document}\n\nSummary: {summary}"
    )
    
    # Preprocess function
    def preprocess_function(examples):
        texts = [
            instruction_template.format(document=doc, summary=summ)
            for doc, summ in zip(examples['document'], examples['summary'])
        ]
        encodings = tokenizer(
            texts, 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
        
        # Clone input_ids for labels and replace padding with -100
        labels = []
        for input_ids in encodings['input_ids']:
            label = [
                (token_id if token_id != tokenizer.pad_token_id else -100)
                for token_id in input_ids
            ]
            labels.append(label)
        
        encodings['labels'] = labels
        return encodings
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    # Training arguments
    model_output_dir = os.path.join(output_dir, 'instruct_model')
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=config['baselines']['instruct']['num_epochs'],
        per_device_train_batch_size=config['baselines']['instruct']['batch_size'],
        per_device_eval_batch_size=config['baselines']['instruct']['batch_size'],
        learning_rate=config['baselines']['instruct']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'] and torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")
    
    print("Instruction tuning completed!")
    
    return model_output_dir