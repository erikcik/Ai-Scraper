import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from scraper.cleanhtml_trial import clean_html
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1") -> tuple:
    """Load the fine-tuned model and tokenizer"""
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    logger.info(f"Loading adapter from: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer

def prepare_prompt(html: str, query: str) -> str:
    """Prepare the prompt in the same format as training data"""
    return f"### Instruction:\nExtract structured information from the HTML content according to this query pattern: {query}\n\n### Input:\n{html}\n\n### Response:\n"

def generate_response(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096-max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:\n")[-1].strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned scraping model")
    parser.add_argument("--html_file", type=str, required=True, help="Path to the HTML file")
    parser.add_argument("--query", type=str, required=True, help="Query pattern for extraction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model name")
    parser.add_argument("--output_file", type=str, help="Optional path to save the output JSON")
    
    args = parser.parse_args()
    
    # Load HTML and clean it
    logger.info(f"Loading and cleaning HTML from {args.html_file}")
    with open(args.html_file, 'r', encoding='utf-8') as f:
        raw_html = f.read()
    cleaned_html = clean_html(raw_html)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    # Prepare prompt and generate response
    prompt = prepare_prompt(cleaned_html, args.query)
    logger.info("Generating response...")
    response = generate_response(model, tokenizer)
    
    # Try to parse response as JSON
    try:
        output = json.loads(response)
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Output saved to {args.output_file}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        logger.error("Failed to parse response as JSON. Raw response:")
        print(response)

if __name__ == "__main__":
    main() 