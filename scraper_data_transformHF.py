import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from tqdm.auto import tqdm
import os
from bs4 import BeautifulSoup, Comment
from typing import List, Dict, Any, Tuple
from scraper.cleanhtml_trial import clean_html  # Make sure this path is correct
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

def fast_chunk_html(clean_html: str, target_chars: int) -> List[str]:
    """
    Extremely fast, approximate HTML chunking.  Prioritizes speed over perfect
    HTML structure preservation.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    # Split by common block-level tags, then by sentences.
    for part in clean_html.split('<p>'):  # Start with paragraph splitting
        for sentence in part.split('. '): # Then split by sentences
            sentence_length = len(sentence)
            if current_length + sentence_length > target_chars and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

    if current_chunk:
        chunks.append(''.join(current_chunk))
    return chunks

def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    input_text: str,
    output_text: str = None
) -> str:
    """Apply chat template to format the conversation consistently"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts structured information from HTML content."},
        {"role": "user", "content": f"{instruction}\n\nInput:\n{input_text}"},
    ]
    
    if output_text:
        messages.append({"role": "assistant", "content": output_text})
    
    # Use ChatML format directly
    conversation = "<|im_start|>system\nYou are a helpful assistant that extracts structured information from HTML content.<|im_end|>\n"
    conversation += f"<|im_start|>user\n{instruction}\n\nInput:\n{input_text}<|im_end|>\n"
    
    if output_text:
        conversation += f"<|im_start|>assistant\n{output_text}<|im_end|>\n"
    else:
        conversation += "<|im_start|>assistant\n"  # For generation
    
    return conversation

def process_item(item: Dict, tokenizer_name: str, max_seq_length: int, chars_per_token: float) -> List[Dict]:
    """Process a single item with chat template"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Make sure we have the right special tokens
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "</s>",
    }
    tokenizer.add_special_tokens({"pad_token": "</s>"})  # Ensure pad token is set
    
    instruction = "Extract structured information from the HTML content according to this query pattern: " + item["query"]
    cleaned_html = clean_html(item["html"])
    target_chars = int((max_seq_length - 1000) * chars_per_token * 0.9)
    chunks = fast_chunk_html(cleaned_html, target_chars)
    output = json.dumps(item["output"], indent=2)
    
    processed_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_instruction = f"{instruction} (Part {chunk_idx + 1}/{len(chunks)})"
        
        formatted_text = apply_chat_template(
            tokenizer=tokenizer,
            instruction=chunk_instruction,
            input_text=chunk,
            output_text=output if chunk_idx == len(chunks)-1 else 'Continue processing next chunk.'
        )
        
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors="pt"
        )
        
        entry = {
            "input_ids": tokenized["input_ids"][0].tolist(),
            "attention_mask": tokenized["attention_mask"][0].tolist(),
            "labels": tokenized["input_ids"][0].tolist()
        }
        processed_chunks.append(entry)
    
    return processed_chunks

def create_training_dataset(json_file_path: str, output_file: str, model_name: str, max_seq_length: int = 4096):
    """Dataset creation with parallel processing and fast chunking."""
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Calculate chars per token ratio for quick estimation
    sample_text = "Sample text for estimation" * 100
    tokens = tokenizer(sample_text, return_tensors="pt")
    chars_per_token = len(sample_text) / len(tokens.input_ids[0])
    print(f"Estimated chars per token: {chars_per_token}")

    print(f"Loading dataset from {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    preprocessed_data = []

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Create partial function with fixed arguments
        process_fn = partial(process_item, tokenizer_name=model_name, max_seq_length=max_seq_length, chars_per_token=chars_per_token)
        
        # Use executor.map for parallel processing of items
        results = list(tqdm(executor.map(process_fn, data), total=len(data), desc="Preprocessing examples"))

        # Flatten the list of lists
        for result in results:
            preprocessed_data.extend(result)

    # Save to JSONL file
    print(f"\nSaving {len(preprocessed_data)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:  # Use 'w' mode since we're writing all at once
        for entry in tqdm(preprocessed_data, desc="Saving to JSONL"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("\nCreating final dataset")
    return Dataset.from_json(output_file)  # Use from_json for consistency

def main():
    print('less goooo')
    parser = argparse.ArgumentParser(description='Preprocess dataset for Axolotl training')
    parser.add_argument('--output_file', type=str, default='web_scraping_dataset.jsonl',
                       help='Output file name for the preprocessed dataset')
    parser.add_argument('--input_file', type=str, default='amazon_gaming_monitors.json',
                       help='Input JSON file containing the raw dataset')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='Model name or path for tokenization')

    args = parser.parse_args()

    # Clear output file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    print(f"Starting preprocessing of {args.input_file}")
    dataset = create_training_dataset(
        args.input_file,
        args.output_file,
        args.model_name
    )
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()